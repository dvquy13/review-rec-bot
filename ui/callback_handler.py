from typing import Any, Dict, List, Optional

from chainlit.context import context_var
from chainlit.element import Text
from chainlit.step import Step, StepType
from literalai.helper import utc_now
from llama_index.core.callbacks import TokenCountingHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

DEFAULT_IGNORE = [
    CBEventType.CHUNKING,
    CBEventType.SYNTHESIZE,
    CBEventType.EMBEDDING,
    CBEventType.NODE_PARSING,
    CBEventType.TREE,
]


class LlamaIndexCallbackHandler(TokenCountingHandler):
    """Base callback handler that can be used to track event starts and ends."""

    steps: Dict[str, Step]

    def __init__(
        self,
        event_starts_to_ignore: List[CBEventType] = DEFAULT_IGNORE,
        event_ends_to_ignore: List[CBEventType] = DEFAULT_IGNORE,
    ) -> None:
        """Initialize the base callback handler."""
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
        )

        self.steps = {}

    def _get_parent_id(self, event_parent_id: Optional[str] = None) -> Optional[str]:
        if event_parent_id and event_parent_id in self.steps:
            return event_parent_id
        elif context_var.get().current_step:
            return context_var.get().current_step.id
        else:
            return None

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        step_type: StepType = "undefined"
        step_name = event_type.value
        step_input = payload or {}
        step_elements = None

        print(f"on_event_start: {event_type=}, {payload=}")
        if event_type == CBEventType.RETRIEVE:
            step_type = "tool"
        elif event_type == CBEventType.QUERY:
            step_type = "tool"
        elif event_type == CBEventType.FUNCTION_CALL:
            step_type = "tool"
            # on_event_start: event_type=<CBEventType.FUNCTION_CALL: 'function_call'>, payload={<EventPayload.FUNCTION_CALL: 'function_call'>: '{"input":"best places to enjoy cold brew coffee"}', <EventPayload.TOOL: 'tool'>: ToolMetadata(de
            # scription='useful for when you want to find restaurants based on end-user reviews. Takes input in a question format, e.g.: What are the best Vietnamese restaurants in Texas?', name='reco_review', fn_schema=<class 'llama_in
            # dex.core.tools.types.DefaultToolFnSchema'>, return_direct=False)}
            tool = payload.get(EventPayload.TOOL)
            tool_name = tool.name
            fn_call_input = payload.get(EventPayload.FUNCTION_CALL)
            step_name = f"tool {tool_name}"
            step_input = fn_call_input
            step_elements = [
                Text(
                    name=step_name,
                    content=str(fn_call_input),
                    display="side",
                )
            ]
        else:
            return event_id

        step = Step(
            name=step_name,
            type=step_type,
            parent_id=self._get_parent_id(parent_id),
            id=event_id,
            elements=step_elements,
        )

        self.steps[event_id] = step
        step.start = utc_now()
        step.input = step_input
        context_var.get().loop.create_task(step.send())
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        step = self.steps.get(event_id, None)

        print(f"on_event_end: {event_type=}, {payload=}")

        if payload is None or step is None:
            return

        step.end = utc_now()

        if event_type == CBEventType.QUERY:
            response = payload.get(EventPayload.RESPONSE)
            source_nodes = getattr(response, "source_nodes", None)
            if source_nodes:
                source_refs = ", ".join(
                    [f"Source {idx}" for idx, _ in enumerate(source_nodes)]
                )
                step.elements = [
                    Text(
                        name=f"Source {idx}",
                        content=source.text or "Empty node",
                        display="side",
                    )
                    for idx, source in enumerate(source_nodes)
                ]
                step.output = f"Retrieved the following sources: {source_refs}"
                context_var.get().loop.create_task(step.update())

        elif event_type == CBEventType.FUNCTION_CALL:
            ...

        elif event_type == CBEventType.RETRIEVE:
            sources = payload.get(EventPayload.NODES)
            if sources:
                source_refs = ", ".join(
                    [f"Source {idx}" for idx, _ in enumerate(sources)]
                )
                step.elements = [
                    Text(
                        name=f"Source {idx}",
                        display="side",
                        content=source.node.get_text() or "Empty node",
                    )
                    for idx, source in enumerate(sources)
                ]
                step.output = f"Retrieved the following sources: {source_refs}"
            context_var.get().loop.create_task(step.update())
        else:
            step.output = payload
            context_var.get().loop.create_task(step.update())

        self.steps.pop(event_id, None)

    def _noop(self, *args, **kwargs):
        pass

    start_trace = _noop
    end_trace = _noop
