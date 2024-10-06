import pyaudio


INPUT_SAMPLE_RATE = 24000  # Input sample rate
INPUT_CHUNK_SIZE = 2048  # Input chunk size
OUTPUT_SAMPLE_RATE = 24000  # Output sample rate. ** Note: This must be 24000 **
OUTPUT_CHUNK_SIZE = 4096  # Output chunk size
STREAM_FORMAT = pyaudio.paInt16  # Stream format
INPUT_CHANNELS = 1  # Input channels
OUTPUT_CHANNELS = 1  # Output channels
OUTPUT_SAMPLE_WIDTH = 2  # Output sample width

INSTRUCTIONS = """Please do a role-play starting now.
# Role-play Setting
- Your name is 'Hanako' and you are a girl.
- Please respond in English.
"""
VOICE_TYPE = "shimmer"  # alloy, echo, shimmer
TEMPERATURE = 0.7
MAX_RESPONSE_OUTPUT_TOKENS = 4096

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
    }
]
TOOL_CHOICE = "auto"

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

TOOL_FUNCTION_LIST = [
    get_your_info
]
TOOL_MAP = {
    tool_info["name"]: tool_func
    for tool_info, tool_func in zip(TOOLS, TOOL_FUNCTION_LIST)
}
