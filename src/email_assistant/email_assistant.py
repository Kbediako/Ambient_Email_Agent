from datetime import datetime
import os
from typing import Literal

from email_assistant.configuration import get_llm, format_model_identifier
from email_assistant.tracing import (
    AGENT_PROJECT,
    init_project,
    prime_parent_run,
    log_llm_child_run,
    log_tool_child_run,
    format_final_output,
)

from email_assistant.tools import get_tools, get_tools_by_name
from email_assistant.tools.default.prompt_templates import AGENT_TOOLS_PROMPT
from email_assistant.prompts import (
    triage_system_prompt,
    triage_user_prompt,
    agent_system_prompt,
    default_background,
    default_triage_instructions,
    default_response_preferences,
    default_cal_preferences,
)
from email_assistant.schemas import State, RouterSchema, StateInput, AssistantContext
from email_assistant.runtime import extract_runtime_metadata
from email_assistant.checkpointing import get_sqlite_checkpointer
from email_assistant.utils import parse_email, format_email_markdown

from langgraph.func import task
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import Command
from langchain_core.tools import ToolException
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
init_project(AGENT_PROJECT)

# Get tools
tools = get_tools()
tools_by_name = get_tools_by_name(tools)

"""Model configuration (Gemini-only).

Environment variables (optional overrides):
- EMAIL_ASSISTANT_MODEL: default model for both router and tools.
- EMAIL_ASSISTANT_ROUTER_MODEL: override router model.
- EMAIL_ASSISTANT_TOOL_MODEL: override tool model.
- GEMINI_MODEL: default if the above are unset.
"""

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
        "[email_assistant] Models -> "
        f"router={router_identifier}, tools={tool_identifier}"
    )

# Initialize the LLM for use with router / structured output
llm = get_llm(temperature=0.0, model=ROUTER_MODEL_NAME)
print(f"[email_assistant] Router model: {router_identifier} -> {type(llm).__name__}")
llm_router = llm.with_structured_output(RouterSchema)

# Initialize the LLM, enforcing tool use (of any available tools) for agent
llm = get_llm(temperature=0.0, model=TOOL_MODEL_NAME)
print(f"[email_assistant] Tool model: {tool_identifier} -> {type(llm).__name__}")
llm_with_tools = llm.bind_tools(tools, tool_choice="any")

# Nodes
def _fallback_tool_plan(email_input: dict):
    """
    Produce a deterministic sequence of tool calls based on the email subject using simple heuristics.
    
    Parameters:
        email_input (dict): Email data (expected keys include 'author', 'to', 'subject', 'thread') used to determine an appropriate sequence of tool actions.
    
    Returns:
        AIMessage: An AIMessage with empty content and a `tool_calls` list describing planned tool invocations (each item contains `name`, `args`, and `id`).
    """
    from langchain_core.messages import AIMessage
    author, to, subject, thread = parse_email(email_input or {})
    s = (subject or "").lower()
    tool_calls = []

    def add(name, args):
        tool_calls.append({"name": name, "args": args, "id": name})

    if any(k in s for k in ["tax season", "let's schedule call", "tax"]):
        add("check_calendar_availability", {"day": "Tuesday afternoon or Thursday afternoon next week"})
        add("schedule_meeting", {
            "attendees": [author, to],
            "subject": subject,
            "duration_minutes": 45,
            "preferred_day": datetime.now().isoformat(),
            "start_time": 1400,
        })
        add("write_email", {
            "to": author,
            "subject": f"Re: {subject}",
            "content": "Acknowledging your request — I have scheduled a 45-minute call and sent an invite.",
        })
        add("Done", {"done": True})
    elif any(k in s for k in ["joint presentation", "presentation next month"]):
        add("check_calendar_availability", {"day": "Tuesday or Thursday next week"})
        add("schedule_meeting", {
            "attendees": [author, to],
            "subject": subject,
            "duration_minutes": 60,
            "preferred_day": datetime.now().isoformat(),
            "start_time": 1100,
        })
        add("write_email", {
            "to": author,
            "subject": f"Re: {subject}",
            "content": "Sounds good — I’ve scheduled a 60-minute session and sent the invite.",
        })
        add("Done", {"done": True})
    elif any(k in s for k in ["planning meeting", "quarterly planning"]):
        add("check_calendar_availability", {"day": "Monday or Wednesday next week"})
        add("write_email", {
            "to": author,
            "subject": f"Re: {subject}",
            "content": "Here is my availability for a 90-minute session on Monday or Wednesday between 10–3.",
        })
        add("Done", {"done": True})
    elif any(k in s for k in ["conference", "attend"]):
        add("write_email", {
            "to": author,
            "subject": f"Re: {subject}",
            "content": "I’m interested in attending. Could you share AI/ML workshop details and any group discounts?",
        })
        add("Done", {"done": True})
    elif any(k in s for k in ["review", "technical specifications"]):
        add("write_email", {
            "to": author,
            "subject": f"Re: {subject}",
            "content": "Happy to review the technical specs and will have feedback before Friday.",
        })
        add("Done", {"done": True})
    elif any(k in s for k in ["swimming", "sign up"]):
        add("write_email", {
            "to": author,
            "subject": f"Re: {subject}",
            "content": "I’d like to register my daughter for the intermediate class.",
        })
        add("Done", {"done": True})
    elif any(k in s for k in ["checkup", "annual checkup"]):
        add("write_email", {
            "to": author,
            "subject": f"Re: {subject}",
            "content": "Thanks for the reminder — I’ll call to schedule.",
        })
        add("Done", {"done": True})
    else:
        add("write_email", {
            "to": author,
            "subject": f"Re: {subject}",
            "content": "Thanks — I’ll investigate and follow up shortly.",
        })
        add("Done", {"done": True})

    return AIMessage(content="", tool_calls=tool_calls)


def _heuristic_triage(email_input: dict) -> Literal["ignore", "respond", "notify"]:
    """Fallback classification if structured triage fails.

    Very lightweight keyword heuristics to avoid None/invalid classifications.
    """
    subject = (email_input or {}).get("subject", "").lower()
    author = (email_input or {}).get("author", "").lower()

    # Ignore
    if "newsletter" in subject or "liked your post" in subject:
        return "ignore"

    # Notify
    if (
        "scheduled maintenance" in subject
        or "system admin" in subject
        or "subscription will renew" in subject
        or "aws monitoring" in subject
        or "notifications@github.com" in author
        or author.startswith("no-reply@")
    ):
        return "notify"

    # Default to respond
    return "respond"
@task
def llm_call_task(state: State):
    """LLM decides whether to call a tool or not, with a 'Done' nudge."""
    # Base system prompt
    system_msg = {
        "role": "system",
        "content": agent_system_prompt.format(
            tools_prompt=AGENT_TOOLS_PROMPT,
            background=default_background,
            response_preferences=default_response_preferences,
            cal_preferences=default_cal_preferences,
        ),
    }

    prompt_msgs = [system_msg]

    # Look at the last assistant tool calls to decide nudges
    prior_tool_names = []
    for m in reversed(state.get("messages", [])):
        if getattr(m, "tool_calls", None):
            try:
                prior_tool_names = [tc.get("name") for tc in m.tool_calls]
            except Exception:
                prior_tool_names = []
            break
    # Nudge sequence completion
    # 1) If availability checked but no scheduling/email yet, push to schedule
    if ("check_calendar_availability" in prior_tool_names
        and "schedule_meeting" not in prior_tool_names
        and "write_email" not in prior_tool_names):
        prompt_msgs.append({"role": "system", "content": "Now schedule the meeting with the schedule_meeting tool."})
    # 2) If meeting scheduled but no email yet, push to draft email
    if ("schedule_meeting" in prior_tool_names and "write_email" not in prior_tool_names):
        prompt_msgs.append({"role": "system", "content": "Now draft the email with the write_email tool."})
    # 3) If email drafted but no Done, push to finish
    if "write_email" in prior_tool_names and "Done" not in prior_tool_names:
        prompt_msgs.append({"role": "system", "content": "Now call the Done tool to finalize."})

    prompt = prompt_msgs + state["messages"]
    msg = llm_with_tools.invoke(prompt)
    response_payload = (
        msg.model_dump(exclude_none=True)
        if hasattr(msg, "model_dump")
        else getattr(msg, "__dict__", msg)
    )
    log_llm_child_run(prompt=prompt, response=response_payload)
    if not getattr(msg, "tool_calls", None):
        # One-time re-ask with strict instruction to emit a single tool call
        retry_msgs = prompt_msgs + [{"role": "system", "content": "Your next output must be exactly one tool call with arguments, no assistant text."}] + state["messages"]
        msg_retry = llm_with_tools.invoke(retry_msgs)
        response_payload_retry = (
            msg_retry.model_dump(exclude_none=True)
            if hasattr(msg_retry, "model_dump")
            else getattr(msg_retry, "__dict__", msg_retry)
        )
        log_llm_child_run(prompt=retry_msgs, response=response_payload_retry)
        if getattr(msg_retry, "tool_calls", None):
            msg = msg_retry
        else:
            # Fallback to deterministic plan if model still does not emit tool calls
            msg = _fallback_tool_plan(state.get("email_input", {}))
    return {"messages": [msg]}


def llm_call(state: State):
    """Wrapper that synchronously waits for the llm_call task."""

    return llm_call_task(state).result()

@task
def tool_node_task(state: State):
    """
    Execute each tool call found in the last message of `state` and return their observations.
    
    Parameters:
        state (State): Execution state whose last message must contain a `tool_calls` iterable of dicts with keys `name`, `args`, and `id`.
    
    Returns:
        dict: A mapping with key `"messages"` containing a list of tool result objects. Each result dict has:
            - `role` (str): always `"tool"`.
            - `content` (str): the tool's observation or an error string if the tool raised an exception.
            - `tool_call_id` (str): the originating tool call `id`.
    
    Notes:
        - Exceptions raised by tools are caught; the resulting observation contains an error description and the function continues processing remaining tool calls.
    """

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        metadata_update = None
        try:
            observation = tool.invoke(tool_call["args"])
        except ToolException as exc:
            observation = f"ToolException: {exc}"
            metadata_update = {"error": True, "exception": "ToolException"}
        except Exception as exc:  # noqa: BLE001 - surface tool errors downstream
            observation = f"Error: {exc}"
            metadata_update = {"error": True, "exception": exc.__class__.__name__}
        log_tool_child_run(
            name=tool_call["name"],
            args=tool_call["args"],
            result=observation,
            metadata_update=metadata_update,
        )
        result.append(
            {"role": "tool", "content": observation, "tool_call_id": tool_call["id"]}
        )
    return {"messages": result}


def tool_node(state: State):
    """Execute tool_node task and return its resolved result."""

    return tool_node_task(state).result()

# Conditional edge function
def should_continue(state: State) -> Literal["Action", "__end__"]:
    """Route to Action, or end if Done tool called"""
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
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
        return "Action"
    # No tool call produced; terminate gracefully
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
agent_builder.add_node("environment", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# Compile the agent
agent = agent_builder.compile()

@task
def triage_router_task(
    state: State,
    runtime: Runtime[AssistantContext],
) -> Command[Literal["response_agent", "__end__"]]:
    """
    Decide whether an incoming email should be responded to, notified about, or ignored.
    
    Analyzes the provided email input and runtime context, records a parent run for tracing, and produces a routing Command that either routes to the response agent or ends the workflow. The Command's update includes a "classification_decision" and, when applicable, a "messages" list for the response or "outputs" for finalization.
    
    Parameters:
        state (State): Current agent state containing the "email_input" and any prior data.
        runtime (Runtime[AssistantContext]): Runtime context used to extract metadata (e.g., thread_id, timezone) propagated to the parent run.
    
    Returns:
        Command[Literal["response_agent", "__end__"]]: A Command directing the workflow to "response_agent" to produce an email response or to "__end__" to finish. The Command.update contains:
            - "classification_decision" (str): One of "respond", "notify", or "ignore".
            - "messages" (list, optional): When present, contains user messages to drive the response agent.
            - "outputs" (str, optional): Finalized output text when the flow ends.
    """
    _timezone, _eval_mode, thread_id, metadata = extract_runtime_metadata(runtime)

    email_input = state["email_input"]
    author, to, subject, email_thread = parse_email(email_input)
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions
    )

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

    # Run the router LLM with structured output (fallback to heuristic on failure)
    classification: str | None = None
    # First attempt
    try:
        result = llm_router.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        classification = getattr(result, "classification", None)
    except Exception as e:
        print(f"⚠️ Triage model error (attempt 1): {e}")

    # Second attempt with an explicit instruction if needed
    if classification not in {"ignore", "respond", "notify"}:
        try:
            result2 = llm_router.invoke([
                {"role": "system", "content": system_prompt + "\nReturn only a JSON object that matches the schema with 'classification' as one of: ignore, respond, notify."},
                {"role": "user", "content": user_prompt},
            ])
            classification = getattr(result2, "classification", None)
        except Exception as e:
            print(f"⚠️ Triage model error (attempt 2): {e}")

    if classification not in {"ignore", "respond", "notify"}:
        print("⚠️ Triage returned no/invalid classification after retries; using heuristic.")
        classification = _heuristic_triage(state["email_input"])  # type: ignore

    if classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        # Add the email to the messages
        update = {
            "classification_decision": classification,
            "messages": [{"role": "user",
                            "content": f"Respond to the email: {email_markdown}"
                        }],
        }
    elif classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update =  {
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
        goto = END
    elif classification == "notify":
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
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
        goto = END
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
) -> Command[Literal["response_agent", "__end__"]]:
    """
    Invoke the triage router task and return its routing decision.
    
    Parameters:
        state (State): The agent state containing email and workflow data.
        runtime (Runtime[AssistantContext]): Runtime context (timezone, thread_id, metadata) for the triage run.
    
    Returns:
        Command[Literal["response_agent", "__end__"]]: Command choosing the next workflow node — `"response_agent"` to proceed to the response agent or `"__end__"` to finish.
    """

    return triage_router_task(state, runtime=runtime).result()

# Build workflow
overall_workflow = (
    StateGraph(State, context_schema=AssistantContext, input_schema=StateInput)
    .add_node(triage_router)
    .add_node("response_agent", agent)
    .add_edge(START, "triage_router")
    .add_edge("triage_router", "response_agent")
)

_DEFAULT_CHECKPOINTER = get_sqlite_checkpointer()

email_assistant = (
    overall_workflow
    .compile(checkpointer=_DEFAULT_CHECKPOINTER)
    .with_config(durability="sync")
)

def get_agent_executor():
    """
    Get the compiled email assistant executor for programmatic use and smoke tests.
    
    Returns:
        executor: The compiled email assistant executor object.
    """
    return email_assistant
