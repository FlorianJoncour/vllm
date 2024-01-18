import os
import json
import jinja2
from enum import Enum
from typing import List, Optional, Union, Any
from vllm.logger import init_logger
from .protocol import (
    ChatCompletionRequest, ChatCompletionToolParam,
    ChoiceDeltaToolCall, ChatCompletionMessageToolCall, Function,
    ChatCompletionAssistantMessage, ChatCompletionToolMessage)

logger = init_logger(__name__)

class ToolsCallsTemplateContext(Enum):
    """ This is used within the template to generate depending on the context. """
    FUNCTIONS_LIST = 1
    CALL_TOKEN = 2
    CALLS_RESPONSES = 3

class ToolsCallsTemplate:
    def __init__(self, template_path=None):
        self.trim_blocks = True
        self.lstrip_blocks = True
        if template_path is None:
            template_path = os.path.dirname(__file__) + "/templates/tools_functions.jinja"
        self.environment = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(template_path)))
        self.template = self.environment.get_template(os.path.basename(template_path))
        self.template.globals['FUNCTIONS_LIST'] = ToolsCallsTemplateContext.FUNCTIONS_LIST
        self.template.globals['CALL_TOKEN'] = ToolsCallsTemplateContext.CALL_TOKEN
        self.template.globals['CALLS_RESPONSES'] = ToolsCallsTemplateContext.CALLS_RESPONSES

    def get_func_call_token(self) -> str:
        """ Return the special token used to find functions calls. """
        return self.template.render(CONTEXT=ToolsCallsTemplateContext.CALL_TOKEN)

    def renderToolResponses(self, tool_calls: [ChatCompletionMessageToolCall]):
        return self.template.render(CONTEXT=ToolsCallsTemplateContext.CALLS_RESPONSES,
                                    tool_calls=tool_calls)

    def renderToolsList(self, tool_choice: Union[str, None], tools_list: [ChatCompletionToolParam]) -> str:
        if isinstance(tool_choice, str) and tool_choice == "auto":
            tool_choice = None
        if tool_choice is not None:
            for tool in tools_list:
                # Search if the tool_choice is in the tools_list
                if tool.type == "function":
                    if tool.function.name == tool_choice:
                        return self.template.render(
                            CONTEXT=ToolsCallsTemplateContext.FUNCTIONS_LIST,
                            tool_choice=tool_choice,
                            tools_list=[tool])
            return None
        else:
            return self.template.render(
                CONTEXT=ToolsCallsTemplateContext.FUNCTIONS_LIST,
                tool_choice=tool_choice,
                tools_list=tools_list)

class OpenAIToolsPrompter:
    """
    https://platform.openai.com/docs/assistants/tools
    """

    def __init__(self, template_path=None):
        self.template = ToolsCallsTemplate(template_path)
        self.call_token_str = self.template.get_func_call_token()
        if self.call_token_str is None:
            logger.error("There is something wrong with the tools template.")
        else:
            self.call_token_pre = self.call_token_str[0]

    def func_call_token_pre(self) -> str:
        return self.call_token_pre

    def func_call_token_size(self) -> int:
        return len(self.call_token_str)

    def func_call_token(self) -> str:
        return self.call_token_str

    def content_from_assistant(self,
                               message: ChatCompletionAssistantMessage) -> str:
        text = self.template.renderToolResponses(message.tool_calls)
        if message.content is None:
            return text
        else:
            return message.content + "\n" + text

    def content_from_tool(self, message: ChatCompletionToolMessage) -> str:
        return message.content

    def inject_prompt(self, request: ChatCompletionRequest):
        """ Generate and inject the prompt for tools calls. """
        if request.tools is not None:
            if self.call_token_str is not None and len(request.tools):
                select_tool_choice = request.tool_choice if (request.tool_choice is not None and request.tool_choice is not "auto") else None
                text_inject = self.template.renderToolsList(tool_choice=select_tool_choice, tools_list=request.tools)
                if isinstance(request.messages, str):
                    request.messages = text_inject + request.messages
                elif isinstance(request.messages,
                                List) and len(request.messages) >= 1:
                    request.messages[
                        0].content = text_inject + request.messages[0].content

class ChatPromptCapture:

    def __init__(self):
        self.content: str = ""
        self.maybe_function_call = False
        self.is_function_call = False
        self.prefix_size = 0
        self.calls_list: List[{}] = []

    def reset(self, reset_calls_list=False):
        self.content = ""
        self.maybe_function_call = False
        self.is_function_call = False
        self.prefix_size = 0
        if reset_calls_list:
            self.calls_list = []

    def num_calls(self):
        return len(self.calls_list)

    def make_calls_list(self, prompter: OpenAIToolsPrompter):
        calls_list = self.content.split(prompter.func_call_token())
        for v_call in calls_list:
            if len(v_call):
                try:
                    call_dict = json.loads(v_call)
                    if "call" in call_dict:
                        self.calls_list.append(call_dict)
                except json.decoder.JSONDecodeError:
                    # Simply ignore invalid functions calls...
                    pass

    def to_ChatCompletionMessageToolCall(
            self, call_id: int) -> Union[ChatCompletionMessageToolCall, None]:
        if len(self.calls_list) and call_id < len(self.calls_list):
            call = self.calls_list[call_id]
            arguments = call["arguments"] if "arguments" in call else None
            function_call = Function(name=call["call"],
                                     arguments=json.dumps(arguments)
                                     if arguments is not None else "")
            return ChatCompletionMessageToolCall(id="call_" + call["call"] +
                                                 "_" + str(call_id),
                                                 type="function",
                                                 function=function_call)
        return None

    def to_ChatCompletionMessageToolCallList(
            self) -> [ChatCompletionMessageToolCall]:
        calls_count = self.num_calls()
        tools_calls_list = []
        for call_id in range(calls_count):
            tools_calls_list.append(
                self.to_ChatCompletionMessageToolCall(
                    call_id=call_id))
        return tools_calls_list

    def to_ChoiceDeltaToolCall(
            self, call_id: int) -> Union[ChoiceDeltaToolCall, None]:
        mesg = self.to_ChatCompletionMessageToolCall(call_id)
        if mesg is not None:
            return ChoiceDeltaToolCall(index=call_id,
                                       id=mesg.id,
                                       type=mesg.type,
                                       function=mesg.function)
        return None

    def to_ChoiceDeltaToolCallList(self):
        calls_count = self.num_calls()
        tools_calls_list = []
        for call_id in range(calls_count):
            tools_calls_list.append(self.to_ChoiceDeltaToolCall(call_id=call_id))
        return tools_calls_list
