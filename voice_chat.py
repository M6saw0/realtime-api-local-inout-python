# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import base64
import json
import os
import sys
import time

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
    Item,
)
from config import (
    INPUT_SAMPLE_RATE,
    INPUT_CHUNK_SIZE,
    OUTPUT_SAMPLE_RATE,
    OUTPUT_CHUNK_SIZE,
    STREAM_FORMAT,
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    OUTPUT_SAMPLE_WIDTH,
    INSTRUCTIONS,
    VOICE_TYPE,
    TEMPERATURE,
    MAX_RESPONSE_OUTPUT_TOKENS,
    TOOLS,
    TOOL_CHOICE,
    TOOL_MAP
)



async def call_tool(client: RTLowLevelClient, previous_item_id: str, call_id: str, tool_name: str, arguments: dict):
    tool_func = TOOL_MAP[tool_name]
    tool_output = tool_func(**arguments)
    print(f"tool_output: {tool_output}")
    await client.send(
        ItemCreateMessage(
            item=FunctionCallOutputItem(
                call_id=call_id, 
                output=tool_output,
            ),
            previous_item_id=previous_item_id,
        )
    )
    await client.send(
        ResponseCreateMessage(
            response=ResponseCreateParams(
            )
        )
    )


async def send_audio(client: RTLowLevelClient, input_stream: pyaudio.Stream):
    def get_audio_data():
        try: 
            return input_stream.read(INPUT_CHUNK_SIZE, exception_on_overflow=False)
        except Exception as e:
            return None
    while not client.closed:
        audio_data = await asyncio.get_event_loop().run_in_executor(None, get_audio_data)
        if audio_data is None:
            continue
        base64_audio = base64.b64encode(audio_data).decode("utf-8")
        await client.send(InputAudioBufferAppendMessage(audio=base64_audio))


async def receive_messages(client: RTLowLevelClient, output_stream: pyaudio.Stream):
    while True:
        message = await client.recv()
        # print(f"{message=}")
        if message is None:
            continue
        match message.type:
            case "session.created":
                # print("Session Created Message")
                # print(f"  Model: {message.session.model}")
                # print(f"  Session Id: {message.session.id}")
                pass
            case "error":
                print("Error Message")
                print(f"  Error: {message.error}")
                pass
            case "input_audio_buffer.committed":
                print("Input Audio Buffer Committed Message")
                print(f"  Item Id: {message.item_id}")
                pass
            case "input_audio_buffer.cleared":
                print("Input Audio Buffer Cleared Message")
                pass
            case "input_audio_buffer.speech_started":
                print("Input Audio Buffer Speech Started Message")
                print(f"  Item Id: {message.item_id}")
                print(f"  Audio Start [ms]: {message.audio_start_ms}")
                pass
            case "input_audio_buffer.speech_stopped":
                print("Input Audio Buffer Speech Stopped Message")
                print(f"  Item Id: {message.item_id}")
                print(f"  Audio End [ms]: {message.audio_end_ms}")
                pass
            case "conversation.item.created":
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
                print("Conversation Item Truncated Message")
                print(f"  Id: {message.item_id}")
                print(f" Content Index: {message.content_index}")
                print(f"  Audio End [ms]: {message.audio_end_ms}")
            case "conversation.item.deleted":
                # print("Conversation Item Deleted Message")
                # print(f"  Id: {message.item_id}")
                pass
            case "conversation.item.input_audio_transcription.completed":
                print("Input Audio Transcription Completed Message")
                print(f"  Id: {message.item_id}")
                print(f"  Content Index: {message.content_index}")
                print(f"  Transcript: {message.transcript}")
            case "conversation.item.input_audio_transcription.failed":
                print("Input Audio Transcription Failed Message")
                print(f"  Id: {message.item_id}")
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
            case "response.done":
                # print("Response Done Message")
                # print(f"  Response Id: {message.response.id}")
                # if message.response.status_details:
                #     print(f"  Status Details: {message.response.status_details.model_dump_json()}")
                # break
                pass
            case "response.output_item.added":
                # print("Response Output Item Added Message")
                # print(f"  Response Id: {message.response_id}")
                # print(f"  Item Id: {message.item.id}")
                pass
            case "response.output_item.done":
                # print("Response Output Item Done Message")
                # print(f"  Response Id: {message.response_id}")
                # print(f"  Item Id: {message.item.id}")
                pass

            case "response.content_part.added":
                # print("Response Content Part Added Message")
                # print(f"  Response Id: {message.response_id}")
                # print(f"  Item Id: {message.item_id}")
                pass
            case "response.content_part.done":
                # print("Response Content Part Done Message")
                # print(f"  Response Id: {message.response_id}")
                # print(f"  ItemPart Id: {message.item_id}")
                pass
            case "response.text.delta":
                # print("Response Text Delta Message")
                # print(f"  Response Id: {message.response_id}")
                # print(f"  Text: {message.delta}")
                pass
            case "response.text.done":
                print("Response Text Done Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Text: {message.text}")
            case "response.audio_transcript.delta":
                # print("Response Audio Transcript Delta Message")
                # print(f"  Response Id: {message.response_id}")
                # print(f"  Item Id: {message.item_id}")
                # print(f"  Transcript: {message.delta}")
                pass
            case "response.audio_transcript.done":
                print("Response Audio Transcript Done Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Item Id: {message.item_id}")
                print(f"  Transcript: {message.transcript}")
            case "response.audio.delta":
                # print("Response Audio Delta Message")
                # print(f"  Response Id: {message.response_id}")
                # print(f"  Item Id: {message.item_id}")
                # print(f"  Audio Data Length: {len(message.delta)}")
                audio_data = base64.b64decode(message.delta)
                # print(f"  Audio Binary Data Length: {len(audio_data)}")
                # audio_duration = len(audio_data) / OUTPUT_SAMPLE_RATE / OUTPUT_SAMPLE_WIDTH / OUTPUT_CHANNELS
                # print(f"  Audio Duration: {audio_duration}")
                # start_time = time.time()
                # audio_data = np.frombuffer(audio_data, dtype=np.int16).tobytes()
                for i in range(0, len(audio_data), OUTPUT_CHUNK_SIZE):
                    await asyncio.get_event_loop().run_in_executor(None, output_stream.write, audio_data[i:i+OUTPUT_CHUNK_SIZE])
                # await asyncio.sleep(max(0, audio_duration - (time.time() - start_time) - 0.05))
                await asyncio.sleep(0)
            case "response.audio.done":
                # print("Response Audio Done Message")
                # print(f"  Response Id: {message.response_id}")
                # print(f"  Item Id: {message.item_id}")
                pass
            case "response.function_call_arguments.delta":
                # print("Response Function Call Arguments Delta Message")
                # print(f"  Response Id: {message.response_id}")
                # print(f"  Arguments: {message.delta}")
                pass
            case "response.function_call_arguments.done":
                print("Response Function Call Arguments Done Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Arguments: {message.arguments}")
                try:
                    arguments = json.loads(message.arguments)
                    await call_tool(client, message.item_id, message.call_id, message.name, arguments)
                except Exception as e:
                    print(f"Error calling tool: {e}")
            case "rate_limits.updated":
                print("Rate Limits Updated Message")
                print(f"  Rate Limits: {message.rate_limits}")
            case _:
                print("Unknown Message")


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
        send_task = asyncio.create_task(send_audio(client, input_stream))
        receive_task = asyncio.create_task(receive_messages(client, output_stream))

        await asyncio.gather(send_task, receive_task)


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

        send_task = asyncio.create_task(send_audio(client, input_stream))
        receive_task = asyncio.create_task(receive_messages(client, output_stream))

        await asyncio.gather(send_task, receive_task)


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
