# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import base64
import json
import os
import sys
import queue
import time
import threading

from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import pyaudio

from rtclient import (
    InputAudioBufferAppendMessage,
    InputAudioTranscription,
    RTLowLevelClient,
    ServerVAD,
    SessionUpdateMessage,
    SessionUpdateParams,
    ResponseCreateMessage,
    ResponseCreateParams,
    UserMessageItem,
    InputTextContentPart,
    ItemCreateMessage,
    FunctionCallOutputItem,
    Item,
)


INPUT_SAMPLE_RATE = 24000  # Input sample rate
INPUT_CHUNK_SIZE = 2048  # Input chunk size
OUTPUT_SAMPLE_RATE = 24000  # Output sample rate. ** Note: This must be 24000 **
OUTPUT_CHUNK_SIZE = 4096  # Output chunk size
STREAM_FORMAT = pyaudio.paInt16  # Stream format
INPUT_CHANNELS = 1  # Input channels
OUTPUT_CHANNELS = 1  # Output channels
OUTPUT_SAMPLE_WIDTH = 2  # Output sample width
VOICE_TYPE = "shimmer"  # alloy, echo, shimmer
TEMPERATURE = 0.7
MAX_RESPONSE_OUTPUT_TOKENS = 4096

def get_your_info(query: str):
    information = (
        "Your name is 'Hanako' and you are a girl."
        "Your age is 20."
        "Your hobby is programming."
        "Your favorite food is sushi."
        "Your favorite color is blue."
        "Your favorite music is J-POP."
        "Your favorite movie is 'Your Name'."
        "Your favorite game is 'League of Legends'."
        "Your favorite sport is baseball."
        "Your favorite team is 'Hanshin Tigers'."
        "Your favorite player is 'Tetsuya Ueki'."
        "Your favorite programming language is Python."
    ) 
    return information

TOOLS = [
    {
        "type": "function",
        "name": "get_your_info",
        "description": "Get the information of your own.(e.g. name, age, hobby, favorite food, favorite color, favorite music, favorite movie, favorite game, favorite sport, favorite team, favorite player, favorite programming language)",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to get the information",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "finish_conversation",
        "description": "Finish the conversation. Used when the conversation is finished.",
        "parameters": {},
    }
]
TOOL_CHOICE = "auto"

INSTRUCTIONS = """Please do a role-play starting now.
# Role-play Setting
- Your name is 'Hanako' and you are a girl.
- Please respond in Japanese.
"""


class DialogueSession:
    def __init__(self, use_azure: bool = True):
        self.use_azure = use_azure
        self.client = None
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.send_task = None
        self.receive_task = None
        self.reset_event = asyncio.Event()
        self.reset_conversation_flag = False

        self.audio_input_queue = queue.Queue()
        self.audio_output_queue = queue.Queue()
        self.execute_tool_queue = asyncio.Queue()
        self.client_event_queue = asyncio.Queue()

    def get_env_var(self, var_name: str) -> str:
        value = os.environ.get(var_name)
        if not value:
            raise OSError(f"Environment variable '{var_name}' is not set or is empty.")
        return value

    async def initialize_streams(self):
        input_default_input_index = self.p.get_default_input_device_info()['index']
        self.input_stream = self.p.open(
            format=STREAM_FORMAT,
            channels=INPUT_CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            output=False,
            frames_per_buffer=INPUT_CHUNK_SIZE,
            input_device_index=input_default_input_index,
            start=False,
        )
        output_default_output_index = self.p.get_default_output_device_info()['index']
        self.output_stream = self.p.open(
            format=STREAM_FORMAT,
            channels=OUTPUT_CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            input=False,
            output=True,
            frames_per_buffer=OUTPUT_CHUNK_SIZE,
            output_device_index=output_default_output_index,
            start=False,
        )
        self.input_stream.start_stream()
        self.output_stream.start_stream()

    async def call_tool(self, previous_item_id: str, call_id: str, tool_name: str, arguments: dict):
        await self.execute_tool_queue.put(
            {
                "previous_item_id": previous_item_id,
                "call_id": call_id,
                "tool_name": tool_name,
                "arguments": arguments
            }
        )


    async def execute_tool(self):
        while True:
            tool_info = await self.execute_tool_queue.get()

            previous_item_id = tool_info["previous_item_id"]
            call_id = tool_info["call_id"]
            tool_name = tool_info["tool_name"]
            arguments = tool_info["arguments"]

            if tool_name == "finish_conversation":
                print("Finishing conversation...")
                await self.client_event_queue.put(
                    ResponseCreateMessage(
                        response=ResponseCreateParams(
                            modalities=["text", "audio"],
                            append_input_items=[
                                UserMessageItem(
                                    content=[
                                        InputTextContentPart(
                                            text="The conversation is about to end. Please conclude the conversation naturally, connecting it to the previous dialogue."
                                        )
                                    ]
                                )
                            ]
                        )
                    )
                )
                await self.reset_session()
                return

            elif tool_name == "get_your_info":
                tool_output = get_your_info(**arguments)
                print(f"Tool: 'get_your_info' called with arguments ({arguments})")
                print(f"  Tool Output: {tool_output}")
                # await self.client_event_queue.put(
                #     ItemCreateMessage(
                #         item=FunctionCallOutputItem(
                #             call_id=call_id, 
                #             output=tool_output,
                #         ),
                #         previous_item_id=previous_item_id,
                #     )
                # )
                await self.client_event_queue.put(
                    ResponseCreateMessage(
                        response=ResponseCreateParams(
                            modalities=["text", "audio"],
                            append_input_items=[
                                UserMessageItem(
                                    content=[
                                        InputTextContentPart(
                                            text=(
                                                "Below is the output of the function. Please use the following information to continue the previous conversation.\n"
                                                "# Function Output\n"
                                                f"{tool_output}"
                                            )
                                        )
                                    ]
                                )
                            ]
                        )
                    )
                )
            else:
                print(f"No such tool: {tool_name}")


    async def send_text_client_event(self):
        while True:
            message = await self.client_event_queue.get()
            await self.client.send(message)


    async def reset_session(self):
        print("Requesting session reset...")
        self.reset_conversation_flag = True

    def listen_audio(self, input_stream: pyaudio.Stream):
        while True:
            audio_data = input_stream.read(INPUT_CHUNK_SIZE, exception_on_overflow=False)
            if audio_data is None:
                continue
            base64_audio = base64.b64encode(audio_data).decode("utf-8")
            self.audio_input_queue.put(base64_audio)

    async def send_audio(self):
        while not self.client.closed:
            base64_audio = await asyncio.get_event_loop().run_in_executor(None, self.audio_input_queue.get)
            await self.client.send(InputAudioBufferAppendMessage(audio=base64_audio))
            await asyncio.sleep(0)

    def play_audio(self, output_stream: pyaudio.Stream):
        while True:
            audio_data = self.audio_output_queue.get()
            output_stream.write(audio_data)

    async def receive_messages(self):
        while True:
            message = await self.client.recv()
            if message is None:
                continue
            match message.type:
                case "session.created":
                    print("Session Created Message")
                case "error":
                    print("Error Message")
                    print(f"  Error: {message.error}")
                case "input_audio_buffer.speech_started":
                    print("Input Audio Buffer Speech Started Message")
                    print(f"  Audio Start [ms]: {message.audio_start_ms}")
                    while not self.audio_output_queue.empty():
                        self.audio_output_queue.get()
                    await asyncio.sleep(0)
                case "input_audio_buffer.speech_stopped":
                    print("Input Audio Buffer Speech Stopped Message")
                    print(f"  Audio End [ms]: {message.audio_end_ms}")
                case "conversation.item.created":
                    print("Conversation Item Created Message")
                    print(f"  Id: {message.item.id}")
                    print(f"  Previous Id: {message.previous_item_id}")
                    if message.item.type == "message":
                        print(f"  Role: {message.item.role}")
                        for index, content in enumerate(message.item.content):
                            print(f"  [{index}]:")
                            print(f"    Content Type: {content.type}")
                            if content.type in ["input_text", "text"]:
                                print(f"    Text: {content.text}")
                            elif content.type in ["input_audio", "audio"]:
                                print(f"    Audio Transcript: {content.transcript}")
                case "conversation.item.truncated":
                    print("Conversation Item Truncated Message")
                    print(f" Content Index: {message.content_index}")
                    print(f"  Audio End [ms]: {message.audio_end_ms}")
                case "conversation.item.deleted":
                    print("Conversation Item Deleted Message")
                case "conversation.item.input_audio_transcription.completed":
                    print("Input Audio Transcription Completed Message")
                    print(f"  Content Index: {message.content_index}")
                    print(f"  Transcript: {message.transcript}")
                case "conversation.item.input_audio_transcription.failed":
                    print("Input Audio Transcription Failed Message")
                    print(f"  Error: {message.error}")
                case "response.created":
                    print("Response Created Message")
                    print(f"  Response Id: {message.response.id}")
                    print("  Output Items:")
                    for index, item in enumerate(message.response.output):
                        print(f"  [{index}]:")
                        print(f"    Item Id: {item.id}")
                        print(f"    Type: {item.type}")
                        if item.type == "message":
                            print(f"    Role: {item.role}")
                            match item.role:
                                case "system":
                                    for content_index, content in enumerate(item.content):
                                        print(f"    [{content_index}]:")
                                        print(f"      Content Type: {content.type}")
                                        print(f"      Text: {content.text}")
                                case "user":
                                    for content_index, content in enumerate(item.content):
                                        print(f"    [{content_index}]:")
                                        print(f"      Content Type: {content.type}")
                                        if content.type == "input_text":
                                            print(f"      Text: {content.text}")
                                        elif content.type == "input_audio":
                                            print(f"      Audio Data Length: {len(content.audio)}")
                                case "assistant":
                                    for content_index, content in enumerate(item.content):
                                        print(f"    [{content_index}]:")
                                        print(f"      Content Type: {content.type}")
                                        print(f"      Text: {content.text}")
                        elif item.type == "function_call":
                            print(f"    Call Id: {item.call_id}")
                            print(f"    Function Name: {item.name}")
                            print(f"    Parameters: {item.arguments}")
                        elif item.type == "function_call_output":
                            print(f"    Call Id: {item.call_id}")
                            print(f"    Output: {item.output}")
                # case "response.text.delta":
                #     print(f"  Text: {message.delta}")
                case "response.text.done":
                    print("Response Text Done Message")
                    print(f"  Text: {message.text}")
                # case "response.audio_transcript.delta":
                #     print(f"{message.delta}", end=" ")
                #     pass
                case "response.audio_transcript.done":
                    print("Response Audio Transcript Done Message")
                    print(f"  Transcript: {message.transcript}")
                    if self.reset_conversation_flag:
                        print("End of conversation.")
                        self.reset_event.set()
                        self.reset_conversation_flag = False
                case "response.audio.delta":
                    print("Response Audio Delta Message")
                    audio_data = base64.b64decode(message.delta)
                    for i in range(0, len(audio_data), OUTPUT_CHUNK_SIZE):
                        self.audio_output_queue.put(audio_data[i:i+OUTPUT_CHUNK_SIZE])
                    await asyncio.sleep(0)
                # case "response.done":
                    # print("Response Done Message")
                case "response.function_call_arguments.done":
                    print("Response Function Call Arguments Done Message")
                    try:
                        arguments = json.loads(message.arguments)
                        await self.call_tool(message.item_id, message.call_id, message.name, arguments)
                    except Exception as e:
                        print(f"Error calling tool: {e}")
                case "rate_limits.updated":
                    print("Rate Limits Updated Message")
                    print(f"  Rate Limits: {message.rate_limits}")
                case _:
                    pass  # Ignore unknown message types.

    async def start_session(self):
        await self.initialize_streams()
        print("Start Processing")
        if self.use_azure:
            endpoint = self.get_env_var("AZURE_OPENAI_ENDPOINT")
            key = self.get_env_var("AZURE_OPENAI_API_KEY")
            deployment = self.get_env_var("AZURE_OPENAI_DEPLOYMENT")
            client_init = RTLowLevelClient(
                endpoint, key_credential=AzureKeyCredential(key), azure_deployment=deployment
            )
            session_params = SessionUpdateParams(
                turn_detection=ServerVAD(type="server_vad"),
                input_audio_transcription=InputAudioTranscription(model="whisper-1"),
                voice=VOICE_TYPE,
                instructions=INSTRUCTIONS,
                temperature=TEMPERATURE,
                max_response_output_tokens=MAX_RESPONSE_OUTPUT_TOKENS,
                tools=TOOLS,
                tool_choice=TOOL_CHOICE,
            )
        else:
            key = self.get_env_var("OPENAI_API_KEY")
            model = self.get_env_var("OPENAI_MODEL")
            client_init = RTLowLevelClient(key_credential=AzureKeyCredential(key), model=model)
            session_params = SessionUpdateParams(
                model=model,
                modalities=["text", "audio"],
                input_audio_format="pcm16",
                output_audio_format="pcm16",
                turn_detection=ServerVAD(type="server_vad", threshold=0.5, prefix_padding_ms=200, silence_duration_ms=200),
                input_audio_transcription=InputAudioTranscription(model="whisper-1"),
                voice=VOICE_TYPE,
                instructions=INSTRUCTIONS,
                temperature=TEMPERATURE,
                max_response_output_tokens=MAX_RESPONSE_OUTPUT_TOKENS,
                tools=TOOLS,
                tool_choice=TOOL_CHOICE,
            )

        async with client_init as client:
            self.client = client
            await self.client.send(SessionUpdateMessage(session=session_params))

            threading.Thread(target=self.listen_audio, args=(self.input_stream,), daemon=True).start()
            threading.Thread(target=self.play_audio, args=(self.output_stream,), daemon=True).start()
            send_audio_task = asyncio.create_task(self.send_audio())
            receive_server_event_task = asyncio.create_task(self.receive_messages())
            execute_tool_task = asyncio.create_task(self.execute_tool())
            send_text_client_event_task = asyncio.create_task(self.send_text_client_event())
            reset_wait_task = asyncio.create_task(self.reset_event.wait())

            # セッションがリセットされるまで待機
            done, pending = await asyncio.wait(
                [
                    send_audio_task, 
                    receive_server_event_task, 
                    execute_tool_task, 
                    send_text_client_event_task, 
                    reset_wait_task
                ],
                return_when=asyncio.FIRST_COMPLETED
            )

            # リセットが要求された場合の処理
            if reset_wait_task in done:
                self.client.close()
                send_audio_task.cancel()
                receive_server_event_task.cancel()
                execute_tool_task.cancel()
                send_text_client_event_task.cancel()
                reset_wait_task.cancel()


                # try:
                #     await asyncio.gather(*pending, return_exceptions=True)
                # except Exception as e:
                #     print(f"Error during session reset: {e}")
                self.reset_event.clear()
                print("Session has been reset. Restarting...")
                await self.start_session()

    async def run(self):
        try:
            await self.start_session()
        except asyncio.CancelledError:
            print("DialogueSession has been cancelled.")
        finally:
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
            self.p.terminate()
            print("Audio streams terminated.")

async def main():
    load_dotenv()
    if len(sys.argv) < 1:
        print("Usage: python dialogue.py <azure|openai>")
        print("If second argument is not provided, it will default to azure")
        sys.exit(1)

    use_azure = True
    if len(sys.argv) == 2 and sys.argv[1] == "openai":
        use_azure = False

    session = DialogueSession(use_azure=use_azure)
    await session.run()

if __name__ == "__main__":
    asyncio.run(main())