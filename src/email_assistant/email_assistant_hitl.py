import os
from typing import Literal

from langgraph.func import task
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
)
from langgraph.types import interrupt, Command
from langchain_core.tools import ToolException

from email_assistant.tools import get_tools, get_tools_by_name
from email_assistant.tools.default.prompt_templates import HITL_TOOLS_PROMPT
from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt_hitl, default_background, default_triage_instructions, default_response_preferences, default_cal_preferences
from email_assistant.configuration import get_llm, format_model_identifier
from email_assistant.schemas import State, RouterSchema, StateInput, AssistantContext
from email_assistant.runtime import extract_runtime_metadata
from email_assistant.utils import parse_email, format_for_display, format_email_markdown
from email_assistant.checkpointing import get_sqlite_checkpointer
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
    Invoke a named tool with the provided arguments, record the run, and return the tool's observation or a readable error message.
    
    Parameters:
        tool_name (str): The registered tool's name to invoke.
        args (dict): Arguments to pass to the tool.
    
    Returns:
        str: The tool's observation text on success, or a human-readable error description if the tool raised an exception.
    """

    tool = tools_by_name[tool_name]
    metadata_update = None
    try:
        observation = tool.invoke(args)
    except ToolException as exc:
        observation = f"ToolException: {exc}"
        metadata_update = {"error": True, "exception": "ToolException"}
    except Exception as exc:  # noqa: BLE001 - ensure tool errors propagate as observations
        observation = f"Error: {exc}"
        metadata_update = {"error": True, "exception": exc.__class__.__name__}
    log_tool_child_run(
        name=tool_name,
        args=args,
        result=observation,
        metadata_update=metadata_update,
    )
    return observation

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
        "[email_assistant_hitl] Models -> "
        f"router={router_identifier}, tools={tool_identifier}"
    )

# Initialize the LLM for use with router / structured output
llm_router = get_llm(temperature=0.0, model=ROUTER_MODEL_NAME).with_structured_output(RouterSchema)

# Initialize the LLM, allowing any tool call (Gemini rejects 'required' as a function name)
llm_with_tools = get_llm(temperature=0.0, model=TOOL_MODEL_NAME).bind_tools(tools, tool_choice="any")

# Nodes 
@task
def triage_router_task(
    state: State,
    runtime: Runtime[AssistantContext],
) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """
    Decide whether an incoming email should be responded to, ignored, or escalated for human notification.
    
    Parses the email from `state["email_input"]`, runs the router model to classify the triage outcome, and updates workflow state and parent run metadata accordingly. Side effects: calls `prime_parent_run` with the email content and metadata; when classification is "ignore" it also primes a final summary output.
    
    Parameters:
        state (State): Current workflow state containing "email_input".
        runtime (Runtime[AssistantContext]): Runtime providing thread and metadata used when priming parent runs.
    
    Returns:
        Command: A command directing the next node; `goto` is one of "triage_interrupt_handler", "response_agent", or "__end__". The `update` contains `classification_decision` and, when the decision is to respond, a `messages` entry with a user prompt to respond to the email.
    """

    _timezone, _eval_mode, thread_id, metadata = extract_runtime_metadata(runtime)

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

    # Format system prompt with background and triage instructions
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions
    )

    # Run the router LLM
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Decision
    classification = result.classification

    # Process the classification decision
    if classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        # Next node
        goto = "response_agent"
        # Update the state
        update = {
            "classification_decision": result.classification,
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

    else:
        if os.getenv("EMAIL_ASSISTANT_TRACE_DEBUG", "").lower() in ("1", "true", "yes"):
            print("⚠️ Invalid triage classification; defaulting to RESPOND.")
        goto = "response_agent"
        update = {
            "classification_decision": "respond",
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email: {email_markdown}",
                }
            ],
        }
    return Command(goto=goto, update=update)


def triage_router(
    state: State,
    runtime: Runtime[AssistantContext],
) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """
    Execute the triage router and return the next workflow command.
    
    Returns:
        Command: A command directing the workflow to "triage_interrupt_handler", "response_agent", or "__end__".
    """

    return triage_router_task(state, runtime=runtime).result()

@task
def triage_interrupt_handler_task(
    state: State,
    _runtime: Runtime[AssistantContext],
) -> Command[Literal["response_agent", "__end__"]]:
    """
    Create a human interrupt to notify and collect a decision about a triaged email, then route the workflow based on the human response.
    
    This function reads `state["email_input"]` and `state["classification_decision"]`, formats an Agent Inbox notification, and sends a HumanInterrupt request. If the interrupt response type is `"response"`, the user's feedback is appended to the returned messages and the command routes to the response agent. If the response type is `"ignore"`, the command routes to the workflow end. The function updates state with a `messages` list suitable for the response agent.
    
    Parameters:
        state (State): Workflow state; must contain `email_input` (raw email text) and `classification_decision` (triage label).
    
    Returns:
        Command[Literal["response_agent", "__end__"]]: A command with `goto` set to either `"response_agent"` or `"__end__"`, and `update` containing the `messages` list.
    
    Raises:
        ValueError: If the interrupt response has an unexpected `type`.
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

    # Agent Inbox responds with a list  
    response = interrupt([request])[0]

    # If user provides feedback, go to response agent and use feedback to respond to email   
    if response["type"] == "response":
        # Add feedback to messages 
        user_input = response["args"]
        # Used by the response agent
        messages.append({"role": "user",
                        "content": f"User wants to reply to the email. Use this feedback to respond: {user_input}"
                        })
        # Go to response agent
        goto = "response_agent"

    # If user ignores email, go to END
    elif response["type"] == "ignore":
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
    runtime: Runtime[AssistantContext],
) -> Command[Literal["response_agent", "__end__"]]:
    """
    Handle a triage human-interrupt and determine the next workflow node.
    
    Returns:
        Command[Literal["response_agent", "__end__"]]: Command directing the workflow to either "response_agent" or "__end__".
    """

    return triage_interrupt_handler_task(state, runtime).result()

@task
def llm_call_task(state: State):
    """
    Invoke the LLM with the HITL system prompt and the current conversation messages.
    
    Builds a prompt by prepending the HITL system prompt (including tool, background, and preference text) to state["messages"], calls the LLM with tool bindings, logs the LLM response payload, and returns the updated messages list.
    
    Returns:
        dict: A mapping with key "messages" whose value is a list containing the LLM response message object.
    """

    prompt = [
        {
            "role": "system",
            "content": agent_system_prompt_hitl.format(
                tools_prompt=HITL_TOOLS_PROMPT,
                background=default_background,
                response_preferences=default_response_preferences,
                cal_preferences=default_cal_preferences,
            ),
        }
    ] + state["messages"]

    msg = llm_with_tools.invoke(prompt)
    response_payload = (
        msg.model_dump(exclude_none=True)
        if hasattr(msg, "model_dump")
        else getattr(msg, "__dict__", msg)
    )
    log_llm_child_run(prompt=prompt, response=response_payload)

    return {"messages": [msg]}


def llm_call(state: State):
    """Synchronously execute the llm_call task."""

    return llm_call_task(state).result()

@task
def interrupt_handler_task(state: State) -> Command[Literal["llm_call", "__end__"]]:
    """
    Handle human-in-the-loop processing for each tool call found in the most recent assistant message.
    
    This creates human interrupts for tool calls that require review (write_email, schedule_meeting, Question), sends them to the Agent Inbox, and translates the chosen human action into messages that update the agent state. Non-HITL tools are executed immediately and their observations are appended. For HITL tools, accepted or edited actions may trigger tool execution and appended tool responses; ignored or response actions produce messages describing the intended instruction for the agent. The function produces an update containing the resulting messages and determines whether the workflow should continue to the "llm_call" node or end.
    
    Parameters:
        state (State): Current workflow state. Expected to include:
            - "messages": a list where the last message may contain `tool_calls` (each with `name`, `args`, and `id`).
            - "email_input": the original email text used to build human interrupt context.
    
    Returns:
        Command[Literal["llm_call", "__end__"]]: A command with:
            - goto: either "llm_call" to continue or "__end__" to finish.
            - update: a dict with a "messages" key containing appended items representing tool results, edited AI messages, or user-feedback artifacts.
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
        response = interrupt([request])[0]

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

            # Update the write_email tool call with the edited content from Agent Inbox
            if tool_call["name"] == "write_email":

                # Execute the tool with edited args
                observation = _invoke_tool_with_logging(
                    tool_call["name"], edited_args
                )

                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})

            # Update the schedule_meeting tool call with the edited content from Agent Inbox
            elif tool_call["name"] == "schedule_meeting":


                # Execute the tool with edited args
                observation = _invoke_tool_with_logging(
                    tool_call["name"], edited_args
                )

                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})
            
            # Catch all other tool calls
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "ignore":
            if tool_call["name"] == "write_email":
                # Don't execute the tool, and tell the agent how to proceed
                result.append({"role": "tool", "content": "User ignored this email draft. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                # Go to END
                goto = END
            elif tool_call["name"] == "schedule_meeting":
                # Don't execute the tool, and tell the agent how to proceed
                result.append({"role": "tool", "content": "User ignored this calendar meeting draft. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                # Go to END
                goto = END
            elif tool_call["name"] == "Question":
                # Don't execute the tool, and tell the agent how to proceed
                result.append({"role": "tool", "content": "User ignored this question. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                # Go to END
                goto = END
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")
            
        elif response["type"] == "response":
            # User provided feedback
            user_feedback = response["args"]
            if tool_call["name"] == "write_email":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append({"role": "tool", "content": f"User gave feedback, which can we incorporate into the email. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            elif tool_call["name"] == "schedule_meeting":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append({"role": "tool", "content": f"User gave feedback, which can we incorporate into the meeting request. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            elif tool_call["name"] == "Question":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append({"role": "tool", "content": f"User answered the question, which can we can use for any follow up actions. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        # Catch all other responses
        else:
            raise ValueError(f"Invalid response: {response}")
            
    # Update the state 
    update = {
        "messages": result,
    }

    return Command(goto=goto, update=update)


def interrupt_handler(state: State) -> Command[Literal["llm_call", "__end__"]]:
    """Synchronously execute the interrupt handler task."""

    return interrupt_handler_task(state).result()

# Conditional edge function
def should_continue(state: State) -> Literal["interrupt_handler", "__end__"]:
    """Route to tool handler, or end if Done tool called"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        names = [tc.get("name") for tc in last_message.tool_calls]
        if "Done" in names:
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

# Add nodes
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

# Build overall workflow
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
email_assistant = (
    overall_workflow
    .compile(checkpointer=_DEFAULT_CHECKPOINTER)
    .with_config(durability="sync")
)
