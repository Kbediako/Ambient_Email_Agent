import os
from typing import Any, Iterable, Literal, cast

from langgraph.func import task
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
)
from langgraph.runtime import Runtime
from langgraph.types import interrupt, Command
from langchain_core.tools import ToolException

from email_assistant.tools import get_tools, get_tools_by_name
from email_assistant.tools.default.prompt_templates import HITL_MEMORY_TOOLS_PROMPT
from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt_hitl_memory, default_triage_instructions, default_background, default_response_preferences, default_cal_preferences, MEMORY_UPDATE_INSTRUCTIONS, MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT
from email_assistant.configuration import get_llm, format_model_identifier
from email_assistant.schemas import (
    AssistantContext,
    State,
    RouterSchema,
    StateInput,
    UserPreferences,
)
from email_assistant.runtime import extract_runtime_metadata
from email_assistant.utils import parse_email, format_for_display, format_email_markdown
from email_assistant.checkpointing import get_sqlite_checkpointer, get_sqlite_store
from email_assistant.tracing import (
    AGENT_PROJECT,
    init_project,
    prime_parent_run,
    log_llm_child_run,
    log_tool_child_run,
    format_final_output,
)
from dotenv import load_dotenv

load_dotenv(".env")
init_project(AGENT_PROJECT)

# Get tools
tools = get_tools(["write_email", "schedule_meeting", "check_calendar_availability", "Question", "Done"])
tools_by_name = get_tools_by_name(tools)


def _invoke_tool_with_logging(tool_name: str, args: dict) -> str:
    """
    Invoke a named tool, record its result as a telemetry child run, and return the tool's observation.
    
    If the tool raises an exception, capture the exception, record error metadata for telemetry, and return a string describing the error.
    
    Returns:
        str: The tool's observation output, or an error string prefixed with `ToolException:` or `Error:` when an exception occurred.
    """

    tool = tools_by_name[tool_name]
    metadata_update = None
    try:
        observation = tool.invoke(args)
    except ToolException as exc:
        observation = f"ToolException: {exc}"
        metadata_update = {"error": True, "exception": "ToolException"}
    except Exception as exc:  # noqa: BLE001 - ensure tool errors reach the LLM loop
        observation = f"Error: {exc}"
        metadata_update = {"error": True, "exception": exc.__class__.__name__}
    log_tool_child_run(
        name=tool_name,
        args=args,
        result=observation,
        metadata_update=metadata_update,
    )
    return observation

# Optional auto-accept for HITL in tests
def _maybe_interrupt(requests):
    """
    Decide whether to synthesize human-in-the-loop actions automatically or defer to normal interrupt handling.
    
    If the environment variable HITL_AUTO_ACCEPT is set to "1", "true", or "yes" (case-insensitive), this function returns a list of synthesized action dictionaries for each request; otherwise it defers to the runtime's standard interrupt handling.
    
    Parameters:
        requests (Iterable[Any] | None): Iterable of interrupt request objects or dicts. Each request may include a `config` with boolean attributes `allow_accept`, `allow_respond`, or `allow_ignore`, and an `action_request` that may contain an `action` string.
    
    Returns:
        list[dict] | Any: When auto-accept is enabled, a list of synthesized actions where each element is a dict with keys:
            - `type`: one of `"accept"`, `"response"`, or `"ignore"`.
            - `args`: additional data for the action (empty dict or response text).
        Otherwise, the result of the normal interrupt processing.
    """

    if os.getenv("HITL_AUTO_ACCEPT", "").lower() not in ("1", "true", "yes"):
        return interrupt(requests)

    def _allow(config: Any, attr: str) -> bool:
        """
        Determine whether a configuration enables a named attribute.
        
        Treats None as disabled. If `config` is a dict, checks the key `attr`; otherwise checks the attribute `attr` on the object and returns its truthiness.
        
        Parameters:
            config (Any): A mapping or object representing configuration; may be None.
            attr (str): The attribute or key name to check.
        
        Returns:
            True if the attribute/key is present and truthy, False otherwise.
        """
        if config is None:
            return False
        if isinstance(config, dict):
            return bool(config.get(attr, False))
        return bool(getattr(config, attr, False))

    synthesized = []
    for req in cast(Iterable[Any] | None, requests) or []:
        config = None
        action = ""
        if isinstance(req, dict):
            config = req.get("config")
            action_request = req.get("action_request") or {}
            action = str(action_request.get("action", ""))
        else:
            config = getattr(req, "config", None)
            action = str(
                getattr(getattr(req, "action_request", None), "action", "")
            )

        if _allow(config, "allow_accept"):
            synthesized.append({"type": "accept", "args": {}})
        elif _allow(config, "allow_respond"):
            responder = "Auto-accept demo: proceed." if action.lower() == "question" else ""
            synthesized.append({"type": "response", "args": responder})
        elif _allow(config, "allow_ignore"):
            synthesized.append({"type": "ignore", "args": {}})
        else:
            synthesized.append({"type": "response", "args": ""})

    return synthesized

# Role-specific model selection (override via env)
DEFAULT_MODEL = (
    os.getenv("EMAIL_ASSISTANT_MODEL")
    or os.getenv("GEMINI_MODEL")
    or "gemini-2.5-pro"
)
ROUTER_MODEL_NAME = os.getenv("EMAIL_ASSISTANT_ROUTER_MODEL") or DEFAULT_MODEL
TOOL_MODEL_NAME = os.getenv("EMAIL_ASSISTANT_TOOL_MODEL") or DEFAULT_MODEL

router_identifier = format_model_identifier(ROUTER_MODEL_NAME)
tool_identifier = format_model_identifier(TOOL_MODEL_NAME)
if os.getenv("EMAIL_ASSISTANT_TRACE_DEBUG", "").lower() in ("1", "true", "yes"):
    print(
        "[email_assistant_hitl_memory] Models -> "
        f"router={router_identifier}, tools={tool_identifier}"
    )

# Initialize the LLM for use with router / structured output
llm_router = get_llm(temperature=0.0, model=ROUTER_MODEL_NAME).with_structured_output(RouterSchema)

# Initialize the LLM, allowing any tool call (Gemini rejects 'required' as a function name)
llm_with_tools = get_llm(temperature=0.0, model=TOOL_MODEL_NAME).bind_tools(tools, tool_choice="any")

def get_memory(store, namespace, default_content=None):
    """Get memory from the store or initialize with default if it doesn't exist.
    
    Args:
        store: LangGraph BaseStore instance to search for existing memory
        namespace: Tuple defining the memory namespace, e.g. ("email_assistant", "triage_preferences")
        default_content: Default content to use if memory doesn't exist
        
    Returns:
        str: The content of the memory profile, either from existing memory or the default
    """
    # Search for existing memory with namespace and key
    user_preferences = store.get(namespace, "user_preferences")
    
    # If memory exists, return its content (the value)
    if user_preferences:
        return user_preferences.value
    
    # If memory doesn't exist, add it to the store and return the default content
    else:
        # Namespace, key, value
        store.put(namespace, "user_preferences", default_content)
        user_preferences = default_content
    
    # Return the default content
    return user_preferences 

def update_memory(store, namespace, messages):
    """Update memory profile in the store with robust fallbacks.
    
    Args:
        store: LangGraph BaseStore instance to update memory
        namespace: Tuple defining the memory namespace, e.g. ("email_assistant", "triage_preferences")
        messages: List of messages to update the memory with
    """

    # Get the existing memory (may be None on first run)
    existing = None
    try:
        existing = store.get(namespace, "user_preferences")
    except Exception:
        existing = None
    current_profile = ""
    if existing is not None:
        try:
            current_profile = existing.value or ""
        except Exception:
            current_profile = str(existing)

    # Try structured update via LLM
    new_profile = None
    try:
        memory_model = os.getenv("EMAIL_ASSISTANT_MEMORY_MODEL") or DEFAULT_MODEL
        llm = get_llm(model=memory_model).with_structured_output(UserPreferences)
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": MEMORY_UPDATE_INSTRUCTIONS.format(
                        current_profile=current_profile, namespace=namespace
                    ),
                }
            ]
            + messages
        )
        # Support model or dict return shapes
        if result is not None and hasattr(result, "user_preferences"):
            new_profile = result.user_preferences
        elif isinstance(result, dict) and "user_preferences" in result:
            new_profile = result["user_preferences"]
    except Exception as e:
        print(f"[memory] LLM update failed: {e}")

    # Fallback: preserve current profile if LLM failed or returned empty
    if not new_profile:
        new_profile = current_profile

    # Save the updated (or preserved) memory to the store
    try:
        store.put(namespace, "user_preferences", new_profile)
    except Exception as e:
        print(f"[memory] Store update failed: {e}")

# Nodes 
@task
def triage_router_task(
    state: State,
    store: BaseStore,
    runtime: Runtime[AssistantContext],
) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """
    Decide whether an incoming email should be responded to, sent for human review, or ignored.
    
    Uses stored triage and response preferences and the router LLM to classify the email, primes the parent run with metadata, and returns a Command directing the workflow. The Command's update includes a "classification_decision" and, when appropriate, a prepared user message for the response agent. If the classification is invalid, the function defaults to "respond".
    
    Returns:
        Command: goto is one of "response_agent", "triage_interrupt_handler", or "__end__"; update contains the fields required by the next node (for example, "classification_decision" and optionally "messages").
    """
    _, _, thread_id, metadata = extract_runtime_metadata(runtime)

    # Parse the email input
    email_input = state["email_input"]
    author, to, subject, email_thread = parse_email(email_input)
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)
    prime_parent_run(
        email_input=email_input,
        email_markdown=email_markdown,
        metadata_update=metadata,
        thread_id=thread_id,
    )

    # Search for existing memory
    triage_instructions = get_memory(
        store, ("email_assistant", "triage_preferences"), default_triage_instructions
    )
    # Also incorporate response preferences to better guide triage decisions
    response_prefs_for_triage = get_memory(
        store, ("email_assistant", "response_preferences"), default_response_preferences
    )
    triage_instructions = (
        f"{triage_instructions}\n\nConsider these user response preferences when deciding notify vs respond:\n{response_prefs_for_triage}"
    )

    # Format system prompt with background and triage instructions
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=triage_instructions,
    )

    # Run the router LLM
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Decision with robust fallback
    classification = getattr(result, "classification", None)
    if classification not in {"ignore", "respond", "notify"}:
        classification = "respond"

    # Process the classification decision
    if classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        # Next node
        goto = "response_agent"
        # Update the state
        update = {
            "classification_decision": classification,
            "messages": [{"role": "user",
                            "content": f"Respond to the email: {email_markdown}"
                        }],
        }
        
    elif classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")

        # Next node
        goto = END
        # Update the state
        update = {
            "classification_decision": classification,
        }
        summary_state = dict(state)
        summary_state["classification_decision"] = classification
        output_text = format_final_output(summary_state)
        prime_parent_run(
            email_input=email_input,
            email_markdown=email_markdown,
            outputs=output_text,
            metadata_update=metadata,
            thread_id=thread_id,
        )

    elif classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information") 

        # Next node
        goto = "triage_interrupt_handler"
        # Update the state
        update = {
            "classification_decision": classification,
        }
        prime_parent_run(
            email_input=email_input,
            email_markdown=email_markdown,
            metadata_update=metadata,
            thread_id=thread_id,
        )

    else:
        raise ValueError(f"Invalid classification: {classification}")
    
    return Command(goto=goto, update=update)


def triage_router(
    state: State,
    store: BaseStore,
    runtime: Runtime[AssistantContext],
) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """
    Run the triage router synchronously to obtain the next workflow command.
    
    Returns:
        Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]: The next node command indicating whether to route to `triage_interrupt_handler`, `response_agent`, or `__end__`.
    """

    return triage_router_task(state, store=store, runtime=runtime).result()

@task
def triage_interrupt_handler_task(
    state: State,
    store: BaseStore,
) -> Command[Literal["response_agent", "__end__"]]:
    """
    Handle a human-in-the-loop interrupt for a triage "notify" decision and route the workflow based on the Agent Inbox response.
    
    Sends a summary of the email to the Agent Inbox as a HumanInterrupt and waits for a single response. If the agent responds with a "response" action, the agent's feedback is appended to the messages, the triage preferences memory is updated to reflect the user's decision to respond, and the workflow is routed to the response agent. If the agent responds with "ignore", the triage preferences memory is updated to reflect the ignore decision and the workflow ends. Any other response type raises a ValueError.
    
    Parameters:
        state (State): Current workflow state; expects keys "email_input" and "classification_decision".
        store (BaseStore): Persistent store used to read/update triage preferences.
    
    Returns:
        Command: A Command whose `goto` is either "response_agent" or "__end__" and whose `update` contains the composed `messages`.
    """
    
    # Parse the email input
    author, to, subject, email_thread = parse_email(state["email_input"])

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    # Create messages
    messages = [{"role": "user",
                "content": f"Email to notify user about: {email_markdown}"
                }]

    # Create interrupt for Agent Inbox
    request: HumanInterrupt = HumanInterrupt(
        action_request=ActionRequest(
            action=f"Email Assistant: {state['classification_decision']}",
            args={},
        ),
        config=HumanInterruptConfig(
            allow_ignore=True,
            allow_respond=True,
            allow_edit=False,
            allow_accept=False,
        ),
        description=email_markdown,
    )

    # Send to Agent Inbox and wait for response
    response = _maybe_interrupt([request])[0]

    # If user provides feedback, go to response agent and use feedback to respond to email   
    if response["type"] == "response":
        # Add feedback to messages 
        user_input = response["args"]
        messages.append({"role": "user",
                        "content": f"User wants to reply to the email. Use this feedback to respond: {user_input}"
                        })
        # Update memory with feedback
        update_memory(store, ("email_assistant", "triage_preferences"), [{
            "role": "user",
            "content": f"The user decided to respond to the email, so update the triage preferences to capture this."
        }] + messages)

        goto = "response_agent"

    # If user ignores email, go to END
    elif response["type"] == "ignore":
        # Make note of the user's decision to ignore the email
        messages.append({"role": "user",
                        "content": f"The user decided to ignore the email even though it was classified as notify. Update triage preferences to capture this."
                        })
        # Update memory with feedback 
        update_memory(store, ("email_assistant", "triage_preferences"), messages)
        goto = END

    # Catch all other responses
    else:
        raise ValueError(f"Invalid response: {response}")

    # Update the state 
    update = {
        "messages": messages,
    }

    return Command(goto=goto, update=update)


def triage_interrupt_handler(
    state: State,
    store: BaseStore,
) -> Command[Literal["response_agent", "__end__"]]:
    """
    Run the triage interrupt handler task and return its routing Command.
    
    Returns:
        Command[Literal["response_agent", "__end__"]]: Command indicating the next step — `"response_agent"` to continue processing or `"__end__"` to finish.
    """

    return triage_interrupt_handler_task(state, store=store).result()

@task
def llm_call_task(state: State, store: BaseStore):
    """
    Determines whether the LLM should invoke a tool and returns the chosen assistant message.
    
    Builds a system prompt incorporating stored calendar and response preferences, calls the tool-enabled LLM to obtain an assistant message (which may include one or more tool calls), retries once with a stricter instruction if no tool call is returned, and falls back to a synthetic "Done" tool call to avoid stalling. Logs LLM interactions and preserves any tool call payload returned by the model.
    
    Returns:
        dict: A mapping with key "messages" whose value is a single-element list containing the assistant message. The message may contain a `tool_calls` field with tool invocation specifications; if the LLM produced no tool calls, the returned message will include a fallback `{"name": "Done", "args": {"done": True}}` tool call.
    """
    
    # Search for existing cal_preferences memory
    cal_preferences = get_memory(store, ("email_assistant", "cal_preferences"), default_cal_preferences)
    
    # Search for existing response_preferences memory
    response_preferences = get_memory(store, ("email_assistant", "response_preferences"), default_response_preferences)

    prompt = [
        {"role": "system", "content": agent_system_prompt_hitl_memory.format(
            tools_prompt=HITL_MEMORY_TOOLS_PROMPT,
            background=default_background,
            response_preferences=response_preferences,
            cal_preferences=cal_preferences,
        )}
    ] + state["messages"]

    # Invoke with a retry and fallback to Done to avoid stalls
    try:
        msg = llm_with_tools.invoke(prompt)
        response_payload = (
            msg.model_dump(exclude_none=True)
            if hasattr(msg, "model_dump")
            else getattr(msg, "__dict__", msg)
        )
        log_llm_child_run(prompt=prompt, response=response_payload)
    except Exception:
        msg = None

    if not getattr(msg, "tool_calls", None):
        retry = [
            {
                "role": "system",
                "content": "Your next output must be exactly one tool call with arguments, no assistant text.",
            }
        ] + prompt
        try:
            msg_retry = llm_with_tools.invoke(retry)
            response_payload_retry = (
                msg_retry.model_dump(exclude_none=True)
                if hasattr(msg_retry, "model_dump")
                else getattr(msg_retry, "__dict__", msg_retry)
            )
            log_llm_child_run(prompt=retry, response=response_payload_retry)
            if getattr(msg_retry, "tool_calls", None):
                msg = msg_retry
        except Exception:
            msg = None
    if not getattr(msg, "tool_calls", None):
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="", tool_calls=[{"name": "Done", "args": {"done": True}, "id": "Done"}])
    return {"messages": [msg]}


def llm_call(state: State, store: BaseStore):
    """Synchronously execute the llm_call task."""

    return llm_call_task(state).result()

@task
def interrupt_handler_task(state: State, store: BaseStore) -> Command[Literal["llm_call", "__end__"]]:
    """
    Request human review for tool calls found in the most recent assistant message and apply the resulting actions.
    
    Sends human interrupt requests for HITL tools ("write_email", "schedule_meeting", "Question"), executes non-HITL tools directly, and incorporates human responses:
    - "accept": executes the original tool call and appends the tool output.
    - "edit": replaces the tool call in the assistant message with the edited version, executes the edited call when applicable, and updates related memories (response or calendar preferences).
    - "ignore": skips execution, appends a note, updates triage preferences, and may end the workflow.
    - "response": records user feedback as a tool message and updates related memories when applicable.
    
    @returns:
        Command with `goto` set to either "llm_call" or "__end__" and `update` containing a `messages` list that reflects executed tool outputs, edited messages, ignore notes, and feedback entries.
    """
    
    # Store messages
    result = []

    # Go to the LLM call node next
    goto = "llm_call"

    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        
        # Allowed tools for HITL
        hitl_tools = ["write_email", "schedule_meeting", "Question"]
        
        # If tool is not in our HITL list, execute it directly without interruption
        if tool_call["name"] not in hitl_tools:

            # Execute search_memory and other tools without interruption
            observation = _invoke_tool_with_logging(
                tool_call["name"], tool_call["args"]
            )
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
            continue
            
        # Get original email from email_input in state
        email_input = state["email_input"]
        author, to, subject, email_thread = parse_email(email_input)
        original_email_markdown = format_email_markdown(subject, author, to, email_thread)
        
        # Format tool call for display and prepend the original email
        tool_display = format_for_display(tool_call)
        description = original_email_markdown + tool_display

        # Configure what actions are allowed in Agent Inbox
        if tool_call["name"] == "write_email":
            config = HumanInterruptConfig(
                allow_ignore=True,
                allow_respond=True,
                allow_edit=True,
                allow_accept=True,
            )
        elif tool_call["name"] == "schedule_meeting":
            config = HumanInterruptConfig(
                allow_ignore=True,
                allow_respond=True,
                allow_edit=True,
                allow_accept=True,
            )
        elif tool_call["name"] == "Question":
            config = HumanInterruptConfig(
                allow_ignore=True,
                allow_respond=True,
                allow_edit=False,
                allow_accept=False,
            )
        else:
            raise ValueError(f"Invalid tool call: {tool_call['name']}")

        # Create the interrupt request
        request: HumanInterrupt = HumanInterrupt(
            action_request=ActionRequest(
                action=tool_call["name"],
                args=tool_call["args"],
            ),
            config=config,
            description=description,
        )

        # Send to Agent Inbox and wait for response
        response = _maybe_interrupt([request])[0]

        # Handle the responses 
        if response["type"] == "accept":

            # Execute the tool with original args
            observation = _invoke_tool_with_logging(
                tool_call["name"], tool_call["args"]
            )
            result.append(
                {"role": "tool", "content": observation, "tool_call_id": tool_call["id"]}
            )
                        
        elif response["type"] == "edit":

            # Preserve the initial tool payload for memory updates
            initial_tool_call = tool_call["args"]
            
            # Get edited args from Agent Inbox
            edited_args = response["args"]["args"]

            # Update the AI message's tool call with edited content (reference to the message in the state)
            ai_message = state["messages"][-1] # Get the most recent message from the state
            current_id = tool_call["id"] # Store the ID of the tool call being edited
            
            # Create a new list of tool calls by filtering out the one being edited and adding the updated version
            # This avoids modifying the original list directly (immutable approach)
            updated_tool_calls = [tc for tc in ai_message.tool_calls if tc["id"] != current_id] + [
                {"type": "tool_call", "name": tool_call["name"], "args": edited_args, "id": current_id}
            ]

            # Create a new copy of the message with updated tool calls rather than modifying the original
            # This ensures state immutability and prevents side effects in other parts of the code
            result.append(ai_message.model_copy(update={"tool_calls": updated_tool_calls}))

            # Save feedback in memory and update the write_email tool call with the edited content from Agent Inbox
            if tool_call["name"] == "write_email":
                
                # Execute the tool with edited args
                observation = _invoke_tool_with_logging(
                    tool_call["name"], edited_args
                )
                
                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})

                # This is new: update the memory
                update_memory(store, ("email_assistant", "response_preferences"), [{
                    "role": "user",
                    "content": f"User edited the email response. Here is the initial email generated by the assistant: {initial_tool_call}. Here is the edited email: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])
            
            # Save feedback in memory and update the schedule_meeting tool call with the edited content from Agent Inbox
            elif tool_call["name"] == "schedule_meeting":
                
                # Execute the tool with edited args
                observation = _invoke_tool_with_logging(
                    tool_call["name"], edited_args
                )
                
                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})

                # This is new: update the memory
                update_memory(store, ("email_assistant", "cal_preferences"), [{
                    "role": "user",
                    "content": f"User edited the calendar invitation. Here is the initial calendar invitation generated by the assistant: {initial_tool_call}. Here is the edited calendar invitation: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])
            
            # Catch all other tool calls
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "ignore":

            if tool_call["name"] == "write_email":
                # Don't execute the tool, and tell the agent how to proceed
                result.append({"role": "tool", "content": "User ignored this email draft. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                # Go to END
                goto = END
                # This is new: update the memory
                update_memory(store, ("email_assistant", "triage_preferences"), state["messages"] + result + [{
                    "role": "user",
                    "content": f"The user ignored the email draft. That means they did not want to respond to the email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])

            elif tool_call["name"] == "schedule_meeting":
                # Don't execute the tool, and tell the agent how to proceed
                result.append({"role": "tool", "content": "User ignored this calendar meeting draft. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                # Go to END
                goto = END
                # This is new: update the memory
                update_memory(store, ("email_assistant", "triage_preferences"), state["messages"] + result + [{
                    "role": "user",
                    "content": f"The user ignored the calendar meeting draft. That means they did not want to schedule a meeting for this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])

            elif tool_call["name"] == "Question":
                # Don't execute the tool, and tell the agent how to proceed
                result.append({"role": "tool", "content": "User ignored this question. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                # Go to END
                goto = END
                # This is new: update the memory
                update_memory(store, ("email_assistant", "triage_preferences"), state["messages"] + result + [{
                    "role": "user",
                    "content": f"The user ignored the Question. That means they did not want to answer the question or deal with this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])

            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "response":
            # User provided feedback
            user_feedback = response["args"]
            if tool_call["name"] == "write_email":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append({"role": "tool", "content": f"User gave feedback, which can we incorporate into the email. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
                # This is new: update the memory
                update_memory(store, ("email_assistant", "response_preferences"), state["messages"] + result + [{
                    "role": "user",
                    "content": f"User gave feedback, which we can use to update the response preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])

            elif tool_call["name"] == "schedule_meeting":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append({"role": "tool", "content": f"User gave feedback, which can we incorporate into the meeting request. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
                # This is new: update the memory
                update_memory(store, ("email_assistant", "cal_preferences"), state["messages"] + result + [{
                    "role": "user",
                    "content": f"User gave feedback, which we can use to update the calendar preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."
                }])

            elif tool_call["name"] == "Question":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append({"role": "tool", "content": f"User answered the question, which can we can use for any follow up actions. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})

            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

    # Update the state 
    update = {
        "messages": result,
    }

    return Command(goto=goto, update=update)


def interrupt_handler(state: State, store: BaseStore) -> Command[Literal["llm_call", "__end__"]]:
    """Synchronously execute the interrupt handler task."""

    return interrupt_handler_task(state).result()

# Conditional edge function
def should_continue(
    state: State,
    runtime: Runtime[AssistantContext],
) -> Literal["interrupt_handler", "__end__"]:
    """
    Decide the next workflow step after the LLM's tool call and finalize parent outputs when finished.
    
    Parameters:
        state (State): Current workflow state containing the message list and email input.
        runtime (Runtime[AssistantContext]): Runtime context used to extract thread and metadata for parent run priming.
    
    Returns:
        str: "interrupt_handler" if there are tool calls that require human-in-the-loop handling, "__end__" if the workflow should terminate (in which case the final output is formatted and the parent run is primed).
    """
    _, _, thread_id, metadata = extract_runtime_metadata(runtime)
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        names = [tc.get("name") for tc in last_message.tool_calls]
        if "Done" in names:
            # TODO: Here, we could update the background memory with the email-response for follow up actions. 
            summary_state = dict(state)
            summary_state.setdefault("classification_decision", "respond")
            output_text = format_final_output(summary_state)
            prime_parent_run(
                email_input=state.get("email_input", {}),
                email_markdown=None,
                outputs=output_text,
                metadata_update=metadata,
                thread_id=thread_id,
            )
            return END
        return "interrupt_handler"
    summary_state = dict(state)
    output_text = format_final_output(summary_state)
    prime_parent_run(
        email_input=state.get("email_input", {}),
        email_markdown=None,
        outputs=output_text,
        metadata_update=metadata,
        thread_id=thread_id,
    )
    return END

# Build workflow
agent_builder = StateGraph(State, context_schema=AssistantContext)

# Add nodes - with store parameter
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("interrupt_handler", interrupt_handler)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "interrupt_handler": "interrupt_handler",
        END: END,
    },
)

# Compile the agent
response_agent = agent_builder.compile()

# Build overall workflow with store and checkpointer
overall_workflow = (
    StateGraph(State, context_schema=AssistantContext, input_schema=StateInput)
    .add_node(triage_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_router")
    .add_edge("triage_router", "response_agent")
    .add_edge("triage_router", "triage_interrupt_handler")
    .add_edge("triage_interrupt_handler", "response_agent")
)

_DEFAULT_CHECKPOINTER = get_sqlite_checkpointer()
_DEFAULT_STORE = get_sqlite_store()

email_assistant = (
    overall_workflow
    .compile(
        checkpointer=_DEFAULT_CHECKPOINTER,
        store=_DEFAULT_STORE,
    )
    .with_config(durability="sync")
)
