class TextContent:
    """文本内容类"""
    def __init__(self, text: str):
        self.text = text


class Content:
    """内容基类"""
    pass


class Message:
    """消息类"""
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
    """作者类"""
    def __init__(self, role, name):
        self.role = role
        self.name = name


class Role:
    """角色枚举类"""
    TOOL = "tool"
    USER = "user" # 没有用到
    ASSISTANT = "assistant" # assistant就是LLM
    AGENT = "agent" 
