class TextContent:
    """Text content class"""
    def __init__(self, text: str):
        self.text = text


class Content:
    """Base content class"""
    pass


class Message:
    """Message class"""
    def __init__(self, author, content, recipient=None, channel=None):
        self.author = author
        self.content = content
        self.recipient = recipient
        self.channel = channel

    def with_recipient(self, recipient):
        self.recipient = recipient
        return self

    def with_channel(self, channel):
        self.channel = channel
        return self


class Author:
    """Author class"""
    def __init__(self, role, name):
        self.role = role
        self.name = name


class Role:
    """Role enum class"""
    TOOL = "tool"
    USER = "user"  # not used
    ASSISTANT = "assistant"  # assistant is the LLM
    AGENT = "agent"
