from typing import Literal
import os

from langgraph.func import task
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
)
from langgraph.types import interrupt, Command

from email_assistant.tools import get_tools, get_tools_by_name
from email_assistant.tools.default.prompt_templates import HITL_MEMORY_TOOLS_PROMPT
from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt_hitl_memory, default_triage_instructions, default_background, default_response_preferences, default_cal_preferences, MEMORY_UPDATE_INSTRUCTIONS, MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT
import logging
from email_assistant.configuration import get_llm
from email_assistant.schemas import State, RouterSchema, StateInput, UserPreferences
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
logger = logging.getLogger(__name__)

# Get tools
tools = get_tools(["write_email", "schedule_meeting", "check_calendar_availability", "Question", "Done"])
tools_by_name = get_tools_by_name(tools)

# Optional auto-accept for HITL in tests
def _maybe_interrupt(requests):
    if os.getenv("HITL_AUTO_ACCEPT", "").lower() in ("1", "true", "yes"):
        return [{"type": "accept", "args": {}}]
    return interrupt(requests)

llm_router = get_llm(temperature=0.0, role="router").with_structured_output(RouterSchema)
llm_with_tools = get_llm(temperature=0.0, role="tool").bind_tools(tools, tool_choice="any")
try:
    router_model = getattr(llm_router, "model", None) or "<resolved>"
    tool_model = getattr(llm_with_tools, "model", None) or "<resolved>"
    logger.info(f"Models → router={router_model}, tools={tool_model}")
except Exception:
    pass

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
        llm = get_llm(role="memory").with_structured_output(UserPreferences)
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
def triage_router_task(state: State, store: BaseStore) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """
    
    # Parse the email input
    email_input = state["email_input"]
    author, to, subject, email_thread = parse_email(email_input)
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)
    prime_parent_run(email_input=email_input, email_markdown=email_markdown)

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
        prime_parent_run(email_input=email_input, email_markdown=email_markdown, outputs=output_text)

    elif classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information") 

        # Next node
        goto = "triage_interrupt_handler"
        # Update the state
        update = {
            "classification_decision": classification,
        }

    else:
        raise ValueError(f"Invalid classification: {classification}")
    
    return Command(goto=goto, update=update)


def triage_router(state: State, store: BaseStore) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """Synchronously execute the triage router task."""

    return triage_router_task(state).result()

@task
def triage_interrupt_handler_task(state: State, store: BaseStore) -> Command[Literal["response_agent", "__end__"]]:
    """Handles interrupts from the triage step"""
    
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
            "content": "The user decided to respond to the email, so update the triage preferences to capture this."
        }] + messages)

        goto = "response_agent"

    # If user ignores email, go to END
    elif response["type"] == "ignore":
        # Make note of the user's decision to ignore the email
        messages.append({"role": "user",
                        "content": "The user decided to ignore the email even though it was classified as notify. Update triage preferences to capture this."
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


def triage_interrupt_handler(state: State, store: BaseStore) -> Command[Literal["response_agent", "__end__"]]:
    """Synchronously execute the triage interrupt handler task."""

    return triage_interrupt_handler_task(state).result()

@task
def llm_call_task(state: State, store: BaseStore):
    """LLM decides whether to call a tool or not"""
    
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
    """Creates an interrupt for human review of tool calls"""
    
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
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            log_tool_child_run(name=tool_call["name"], args=tool_call["args"], result=observation)
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
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            log_tool_child_run(name=tool_call["name"], args=tool_call["args"], result=observation)
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
                        
        elif response["type"] == "edit":

            # Tool selection 
            tool = tools_by_name[tool_call["name"]]
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
                observation = tool.invoke(edited_args)
                log_tool_child_run(name=tool_call["name"], args=edited_args, result=observation)
                
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
                observation = tool.invoke(edited_args)
                log_tool_child_run(name=tool_call["name"], args=edited_args, result=observation)
                
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
def should_continue(state: State, store: BaseStore) -> Literal["interrupt_handler", "__end__"]:
    """Route to tool handler, or end if Done tool called"""
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
            )
            return END
        return "interrupt_handler"
    summary_state = dict(state)
    output_text = format_final_output(summary_state)
    prime_parent_run(
        email_input=state.get("email_input", {}),
        email_markdown=None,
        outputs=output_text,
    )
    return END

# Build workflow
agent_builder = StateGraph(State)

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
    StateGraph(State, input_schema=StateInput)
    .add_node(triage_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_router")
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
