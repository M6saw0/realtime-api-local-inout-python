# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import base64
import json
import os
import queue
import sys
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
    ItemCreateMessage,
    FunctionCallOutputItem,
)
from config import (
    INPUT_SAMPLE_RATE,
    INPUT_CHUNK_SIZE,
    OUTPUT_SAMPLE_RATE,
    OUTPUT_CHUNK_SIZE,
    STREAM_FORMAT,
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    INSTRUCTIONS,
    VOICE_TYPE,
    TEMPERATURE,
    MAX_RESPONSE_OUTPUT_TOKENS,
    TOOLS,
    TOOL_CHOICE,
    TOOL_MAP
)
from logger import Logger

logger = Logger()
audio_input_queue = queue.Queue()
audio_output_queue = queue.Queue()
execute_tool_queue = asyncio.Queue()
client_event_queue = asyncio.Queue()


async def call_tool(client: RTLowLevelClient, previous_item_id: str, call_id: str, tool_name: str, arguments: dict):
    await execute_tool_queue.put(
        {
            "previous_item_id": previous_item_id,
            "call_id": call_id,
            "tool_name": tool_name,
            "arguments": arguments
        }
    )

async def execute_tool(client: RTLowLevelClient):
    while True:
        message = await execute_tool_queue.get()

        previous_item_id = message["previous_item_id"]
        call_id = message["call_id"]
        tool_name = message["tool_name"]
        arguments = message["arguments"]
        tool_func = TOOL_MAP[tool_name]
        tool_output = tool_func(**arguments)
        print(f"tool_output: {tool_output}")
        await logger.info("Client | conversation.item.create (function_call_output)")
        await client_event_queue.put(
            ItemCreateMessage(
                item=FunctionCallOutputItem(
                    call_id=call_id, 
                    output=tool_output,
                ),
                previous_item_id=previous_item_id,
            )
        )
        await logger.info("Client | response.create")
        await client_event_queue.put(
            ResponseCreateMessage(
                response=ResponseCreateParams(
                )
            )
        )

async def send_text_client_event(client: RTLowLevelClient):
    while True:
        message = await client_event_queue.get()
        await client.send(message)


def listen_audio(input_stream: pyaudio.Stream):
    while True:
        audio_data = input_stream.read(INPUT_CHUNK_SIZE, exception_on_overflow=False)
        if audio_data is None:
            continue
        base64_audio = base64.b64encode(audio_data).decode("utf-8")
        audio_input_queue.put(base64_audio)

async def send_audio(client: RTLowLevelClient):
    while not client.closed:
        base64_audio = await asyncio.get_event_loop().run_in_executor(None, audio_input_queue.get)
        await logger.info("Client | input_audio_buffer.append")
        await client.send(InputAudioBufferAppendMessage(audio=base64_audio))
        await asyncio.sleep(0)

def play_audio(output_stream: pyaudio.Stream):
    while True:
        audio_data = audio_output_queue.get()
        output_stream.write(audio_data)

async def receive_messages(client: RTLowLevelClient):
    while True:
        message = await client.recv()
        # print(f"{message=}")
        if message is None:
            continue
        match message.type:
            case "session.created":
                await logger.info(f"Server | session.created | model: {message.session.model}, session_id: {message.session.id}")
            case "error":
                await logger.info(f"Server | error | error message:{message.error}")
            case "input_audio_buffer.committed":
                await logger.info(f"Server | input_audio_buffer.committed | item_id:{message.item_id}")
                pass
            case "input_audio_buffer.cleared":
                await logger.info(f"Server | input_audio_buffer.cleared | item_id: {message.item_id}")
                print("Input Audio Buffer Cleared Message")
                pass
            case "input_audio_buffer.speech_started":
                await logger.info(f"Server | input_audio_buffer.speech_started | item_id: {message.item_id}, audio_start_ms: {message.audio_start_ms}")
                print("Input Audio Buffer Speech Started Message")
                print(f"  Item Id: {message.item_id}")
                print(f"  Audio Start [ms]: {message.audio_start_ms}")
                while not audio_output_queue.empty():
                    audio_output_queue.get()
                await asyncio.sleep(0)
            case "input_audio_buffer.speech_stopped":
                await logger.info(f"Server | input_audio_buffer.speech_stopped | item_id: {message.item_id}, audio_end_ms: {message.audio_end_ms}")
                print("Input Audio Buffer Speech Stopped Message")
                print(f"  Item Id: {message.item_id}")
                print(f"  Audio End [ms]: {message.audio_end_ms}")
                pass
            case "conversation.item.created":
                await logger.info(f"Server | conversation.item.created | item_id: {message.item.id}, previous_item_id: {message.previous_item_id}")
                print("Conversation Item Created Message")
                print(f"  Id: {message.item.id}")
                print(f"  Previous Id: {message.previous_item_id}")
                if message.item.type == "message":
                    print(f"  Role: {message.item.role}")
                    for index, content in enumerate(message.item.content):
                        print(f"  [{index}]:")
                        print(f"    Content Type: {content.type}")
                        if content.type == "input_text" or content.type == "text":
                            print(f"  Text: {content.text}")
                        elif content.type == "input_audio" or content.type == "audio":
                            print(f"  Audio Transcript: {content.transcript}")
                pass
            case "conversation.item.truncated":
                await logger.info(f"Server | conversation.item.truncated | item_id: {message.item_id}, content_index: {message.content_index}, audio_end_ms: {message.audio_end_ms}")
                print("Conversation Item Truncated Message")
                print(f"  Id: {message.item_id}")
                print(f" Content Index: {message.content_index}")
                print(f"  Audio End [ms]: {message.audio_end_ms}")
            case "conversation.item.deleted":
                await logger.info(f"Server | conversation.item.deleted | item_id: {message.item_id}")
                print("Conversation Item Deleted Message")
                print(f"  Id: {message.item_id}")
                pass
            case "conversation.item.input_audio_transcription.completed":
                await logger.info(f"Server | conversation.item.input_audio_transcription.completed | item_id: {message.item_id}, content_index: {message.content_index}, transcript: {message.transcript}")
                print("Input Audio Transcription Completed Message")
                print(f"  Id: {message.item_id}")
                print(f"  Content Index: {message.content_index}")
                print(f"  Transcript: {message.transcript}")
            case "conversation.item.input_audio_transcription.failed":
                await logger.info(f"Server | conversation.item.input_audio_transcription.failed | item_id: {message.item_id}, error: {message.error}")
                print("Input Audio Transcription Failed Message")
                print(f"  Id: {message.item_id}")
                print(f"  Error: {message.error}")
            case "response.created":
                await logger.info(f"Server | response.created | response_id: {message.response.id}")
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
            case "response.done":
                await logger.info(f"Server | response.done | response_id: {message.response.id}")
                # print("Response Done Message")
                # print(f"  Response Id: {message.response.id}")
                # if message.response.status_details:
                #     print(f"  Status Details: {message.response.status_details.model_dump_json()}")
                pass
            case "response.output_item.added":
                await logger.info(f"Server | response.output_item.added | response_id: {message.response_id}, item_id: {message.item.id}")
            case "response.output_item.done":
                await logger.info(f"Server | response.output_item.done | response_id: {message.response_id}, item_id: {message.item.id}")
            case "response.content_part.added":
                await logger.info(f"Server | response.content_part.added | response_id: {message.response_id}, item_id: {message.item_id}")
            case "response.content_part.done":
                await logger.info(f"Server | response.content_part.done | response_id: {message.response_id}, item_id: {message.item_id}")
            case "response.text.delta":
                await logger.info(f"Server | response.text.delta | response_id: {message.response_id}, item_id: {message.item_id}, text: {message.delta}")
            case "response.text.done":
                await logger.info(f"Server | response.text.done | response_id: {message.response_id}, item_id: {message.item_id}, text: {message.text}")
                print("Response Text Done Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Text: {message.text}")
            case "response.audio_transcript.delta":
                await logger.info(f"Server | response.audio_transcript.delta | response_id: {message.response_id}, item_id: {message.item_id}, transcript: {message.delta}")
            case "response.audio_transcript.done":
                await logger.info(f"Server | response.audio_transcript.done | response_id: {message.response_id}, item_id: {message.item_id}, transcript: {message.transcript}")
                print("Response Audio Transcript Done Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Item Id: {message.item_id}")
                print(f"  Transcript: {message.transcript}")
            case "response.audio.delta":
                await logger.info(f"Server | response.audio.delta | response_id: {message.response_id}, item_id: {message.item_id}, audio_data_length: {len(message.delta)}")
                # print("Response Audio Delta Message")
                # print(f"  Response Id: {message.response_id}")
                # print(f"  Item Id: {message.item_id}")
                # print(f"  Audio Data Length: {len(message.delta)}")
                audio_data = base64.b64decode(message.delta)
                # print(f"  Audio Binary Data Length: {len(audio_data)}")
                # audio_duration = len(audio_data) / OUTPUT_SAMPLE_RATE / OUTPUT_SAMPLE_WIDTH / OUTPUT_CHANNELS
                # print(f"  Audio Duration Time: {audio_duration}")
                for i in range(0, len(audio_data), OUTPUT_CHUNK_SIZE):
                    audio_output_queue.put(audio_data[i:i+OUTPUT_CHUNK_SIZE])
                await asyncio.sleep(0)
            case "response.audio.done":
                await logger.info(f"Server | response.audio.done | response_id: {message.response_id}, item_id: {message.item_id}")
            case "response.function_call_arguments.delta":
                await logger.info(f"Server | response.function_call_arguments.delta | response_id: {message.response_id}, item_id: {message.item_id}, arguments: {message.delta}")
            case "response.function_call_arguments.done":
                await logger.info(f"Server | response.function_call_arguments.done | response_id: {message.response_id}, item_id: {message.item_id}, arguments: {message.arguments}")
                try:
                    arguments = json.loads(message.arguments)
                    await call_tool(client, message.item_id, message.call_id, message.name, arguments)
                except Exception as e:
                    print(f"Error calling tool: {e}")
            case "rate_limits.updated":
                await logger.info(f"Server | rate_limits.updated | rate_limits: {message.rate_limits}")
            case _:
                await logger.info(f"Server | {message.type}")


def get_env_var(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise OSError(f"Environment variable '{var_name}' is not set or is empty.")
    return value


async def with_azure_openai():
    endpoint = get_env_var("AZURE_OPENAI_ENDPOINT")
    key = get_env_var("AZURE_OPENAI_API_KEY")
    deployment = get_env_var("AZURE_OPENAI_DEPLOYMENT")

    p = pyaudio.PyAudio()
    input_default_input_index = p.get_default_input_device_info()['index']
    input_stream = p.open(
        format=STREAM_FORMAT,
        channels=INPUT_CHANNELS,
        rate=INPUT_SAMPLE_RATE,
        input=True,
        output=False,
        frames_per_buffer=INPUT_CHUNK_SIZE,
        input_device_index=input_default_input_index,
        start=False,
    )
    output_default_output_index = p.get_default_output_device_info()['index']
    output_stream = p.open(
        format=STREAM_FORMAT,
        channels=OUTPUT_CHANNELS,
        rate=OUTPUT_SAMPLE_RATE,
        input=False,
        output=True,
        frames_per_buffer=OUTPUT_CHUNK_SIZE,
        output_device_index=output_default_output_index,
        start=False,
    )
    input_stream.start_stream()
    output_stream.start_stream()

    print("Start Processing")
    async with RTLowLevelClient(
        endpoint, key_credential=AzureKeyCredential(key), azure_deployment=deployment
    ) as client:
        await logger.info("Client | session.update")
        await client.send(
            SessionUpdateMessage(
                session=SessionUpdateParams(
                    turn_detection=ServerVAD(type="server_vad"),
                    input_audio_transcription=InputAudioTranscription(model="whisper-1"),
                    voice=VOICE_TYPE,
                    instructions=INSTRUCTIONS,
                    temperature=TEMPERATURE,
                    max_response_output_tokens=MAX_RESPONSE_OUTPUT_TOKENS,
                    tools=TOOLS,
                    tool_choice=TOOL_CHOICE,
                )
            )
        )
        threading.Thread(target=listen_audio, args=(input_stream,), daemon=True).start()
        threading.Thread(target=play_audio, args=(output_stream,), daemon=True).start()
        send_task = asyncio.create_task(send_audio(client))
        receive_task = asyncio.create_task(receive_messages(client))
        execute_tool_task = asyncio.create_task(execute_tool(client))
        send_text_client_event_task = asyncio.create_task(send_text_client_event(client))

        await asyncio.gather(send_task, receive_task, execute_tool_task, send_text_client_event_task)



async def with_openai():
    key = get_env_var("OPENAI_API_KEY")
    model = get_env_var("OPENAI_MODEL")

    p = pyaudio.PyAudio()
    input_default_input_index = p.get_default_input_device_info()['index']
    input_stream = p.open(
        format=STREAM_FORMAT,
        channels=INPUT_CHANNELS,
        rate=INPUT_SAMPLE_RATE,
        input=True,
        output=False,
        frames_per_buffer=INPUT_CHUNK_SIZE,
        input_device_index=input_default_input_index,
        start=False,
    )
    output_default_output_index = p.get_default_output_device_info()['index']
    output_stream = p.open(
        format=STREAM_FORMAT,
        channels=OUTPUT_CHANNELS,
        rate=OUTPUT_SAMPLE_RATE,
        input=False,
        output=True,
        frames_per_buffer=OUTPUT_CHUNK_SIZE,
        output_device_index=output_default_output_index,
        start=False,
    )
    input_stream.start_stream()
    output_stream.start_stream()

    print("Start Processing")
    async with RTLowLevelClient(key_credential=AzureKeyCredential(key), model=model) as client:
        await logger.info("Client | session.update")
        await client.send(
            SessionUpdateMessage(
                session=SessionUpdateParams(
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
            )
        )
        print(
            SessionUpdateMessage(
                session=SessionUpdateParams(
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
            )
        )

        threading.Thread(target=listen_audio, args=(input_stream,), daemon=True).start()
        threading.Thread(target=play_audio, args=(output_stream,), daemon=True).start()
        send_task = asyncio.create_task(send_audio(client))
        receive_task = asyncio.create_task(receive_messages(client))
        execute_tool_task = asyncio.create_task(execute_tool(client))
        send_text_client_event_task = asyncio.create_task(send_text_client_event(client))

        await asyncio.gather(send_task, receive_task, execute_tool_task, send_text_client_event_task)


if __name__ == "__main__":
    load_dotenv()
    if len(sys.argv) < 1:
        print("Usage: python dialogue.py <azure|openai>")
        print("If second argument is not provided, it will default to azure")
        sys.exit(1)

    if len(sys.argv) == 2 and sys.argv[1] == "openai":
        asyncio.run(with_openai())
    else:
        asyncio.run(with_azure_openai())
