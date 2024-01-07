from pypi_search.search import find_packages
from openai import OpenAI
import openai
import os
import time
from pygments import highlight
from pygments.lexers import guess_lexer
from pygments.lexers import MarkdownLexer, PythonLexer
from pygments.lexers import CLexer, CppLexer
from pygments.formatters import TerminalFormatter
from pygments.styles import get_style_by_name
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from openai.types.beta.threads.thread_message import ThreadMessage
from openai.types.beta.assistant import Assistant

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import FileHistory
from prompt_toolkit import PromptSession
from prompt_toolkit.filters import is_searching
import subprocess
from datetime import datetime as dt
from prompt_toolkit.completion import NestedCompleter, WordCompleter
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.completion import ThreadedCompleter, Completion
from prompt_toolkit.document import Document
import rich.markup


class SaveFileCompleter(NestedCompleter):
    def __init__(self, files):
        self.files = files

    def get_completions(self, document, complete_event):
        # Get the text before the cursor
        text = document.text_before_cursor
        stripped_len = len(document.text_before_cursor) - len(text)
        if " " in text:
            first_term = text.split(" ")[0]
            pathcomp = PathCompleter()
            remaining = text[len(first_term):].lstrip()
            move_cursor = len(text) - len(remaining) + stripped_len

            new_document = Document(
                remaining,
                cursor_position=document.cursor_position - move_cursor,
            )
            yield from pathcomp.get_completions(new_document, complete_event)

        for file in self.files:
            if file.id.startswith(text):
                yield Completion(
                    file.id,
                    start_position=0,
                    display_meta=file.filename
                )


class FileIdCompleter(ThreadedCompleter):
    def __init__(self, files):
        self.files = files

    def get_completions(self, document, complete_event):
        # Get the text before the cursor
        text_before_cursor = document.text_before_cursor

        # Check if the text starts with '!attach'
        for file in self.files:
            if file.id.startswith(text_before_cursor):
                yield Completion(
                    file.id,
                    start_position=0,
                    display_meta=file.filename
                )


class RunResult:
    STATE_IN_PROGRESS = 1
    STATE_DONE = 2
    STATE_FAILED = 3

    def __init__(self, run):
        self.run = run


class GptRun:
    def __init__(self, client: OpenAI, thread_id: str, agent_id: str):
        self.client = client
        self.thread_id = thread_id
        self.agent_id = agent_id
        self.files = []
        self.citations = []
        self.function = None
        self.func_args = {}
        self.console = Console()

        self.run: openai.types.beta.threads.runs.Run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.agent_id,
        )

    def get_status(self):
        return self.client.beta.threads.runs.retrieve(
            thread_id=self.thread_id,
            run_id=self.run.id,
        ).status

    def collect_steps(self):
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.thread_id,
            run_id=self.run.id
        )
        steps = []
        for d in (run_steps.data):
            if (d.type != "tool_calls"):
                continue
            if (d.step_details.type != 'tool_calls'):
                continue
            for tool in d.step_details.tool_calls:
                tool: openai.types.beta.threads.runs.code_tool_call.CodeToolCall = tool
                openai.types.beta.threads.runs.code_tool_call.CodeToolCall
                # Hack to fix the function dict conversion (missing output)
                if getattr(tool, 'type', None) is None:
                    if "function" in tool:
                        tool["function"]["output"] = ""
                        tool = openai.types.beta.threads.runs.function_tool_call.FunctionToolCall(**tool)

                if (tool.type == "function"):
                    if (tool.function.arguments):
                        self.function = (tool.id, tool.function.name)
                        self.func_args = tool.function.arguments

                if (tool.type != 'code_interpreter'):
                    continue

                if (not tool.code_interpreter.input):
                    continue
                steps.append((
                    tool.code_interpreter.input,
                    tool.code_interpreter.outputs
                ))
        steps.reverse()
        return steps


class GptAgent:
    def __init__(
        self,
        client: OpenAI,
        agent: Assistant,
        thread_id: str | None = None
    ):
        self.console = Console()
        self.client = client
        self.agent = agent
        self.files = self.client.files.list().data
        if thread_id:
            self.thread = self.client.beta.threads.retrieve(thread_id)
        else:
            self.thread = self.client.beta.threads.create()

    def __add_message(self, content: str) -> ThreadMessage | None:
        try:
            return self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=content,
                file_ids=[file.id for file in self.files]
            )
        except openai.APIStatusError as e:
            self.console.print(f"[bold red]{e.body['message']}")
            return None

    def __create_run(self) -> GptRun:
        return GptRun(
            client=self.client,
            thread_id=self.thread.id,
            agent_id=self.agent.id,
        )

    def request(self, content: str) -> GptRun:
        self.__add_message(content)
        return self.__create_run()

    def id(self) -> str:
        return self.agent.id

    def messages(self):
        return self.client.beta.threads.messages.list(
            thread_id=self.thread.id
        )

    def message(self, message_id: str):
        return self.client.beta.threads.messages.retrieve(
            thread_id=self.thread.id,
            message_id=message_id
        )

    def add_file(self, file):
        self.files.append(file)


class GptAgentFactory:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # delete all assistants:
        # for agent in self.client.beta.assistants.list().data:
        #     print(agent)
        #     self.client.beta.assistants.delete(agent)
        # # delete all files:
        # for file in self.client.files.list(purpose="assistants").data:
        #     print(file)
        #     self.client.files.delete(file)

        self.agents = self.client.beta.assistants.list()
        self.console = Console()

    def get_agents(self):
        return self.agents

    def get_agent(
        self,
        id: str,
        thread_id: str | None = None,
    ) -> GptAgent | None:

        agents = self.client.beta.assistants.list()
        for agent in agents.data:
            if agent.id == id:
                return GptAgent(self.client, agent, thread_id=thread_id)
        return None

    def create_agent(
            self,
            name: str,
            instruction: str,
            thread_id: str | None = None,
    ) -> GptAgent:
        agent = self.client.beta.assistants.create(
            name=name,
            instructions=instruction,
            tools=[
                {"type": "code_interpreter"},
                {"type": "retrieval"},
                {"type": "function", "function": {
                    "name": "send_email",
                    "description": "Send an email to the recipient",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipient": {
                                "type": "string",
                                "description": "The recipient email address"
                            },
                            "subject": {
                                "type": "string",
                                "description": "The email subject"
                            },
                            "body": {
                                "type": "string",
                                "description": "Body of the email"
                            }
                        }
                    }
                }}
            ],
            model="gpt-4-1106-preview",
        )
        return GptAgent(self.client, agent, thread_id=thread_id)


class ChatApp(object):
    CITATION_MESSAGE = "Citations are available, you can check them with !citations"
    DEFAULT_INSTRUCTION = "You are Doc, a personal assistant. You can write code and execute it to to help the user. You can also help the user using your knowledge base, you can send email with the function tool"

    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.citations = dict()
        self.received_files = list()
        self.console = Console()
        self.assistant = self.__build_assistant()
        self.new_files = False
        self.history = FileHistory(os.path.join(os.path.expanduser("~"), ".gpt_history"))
        self.session = PromptSession(history=self.history, complete_while_typing=True)

    def __build_assistant(self) -> GptAgent:
        factory = GptAgentFactory()
        id_file = os.path.join(os.path.expanduser("~"), ".gpt_assistant_id")
        agent = None

        if os.path.exists(id_file):
            id = open(id_file).read().strip()
            agent = factory.get_agent(id)

        if agent is None:
            agent = factory.create_agent(
                name="Doc",
                instruction=self.DEFAULT_INSTRUCTION,
            )

            open(id_file, "w").write(agent.id())
        return agent

    def __prompt(self) -> str:
        kb = KeyBindings()

        @kb.add('escape', 'enter', filter=~is_searching)
        def _(event):
            event.current_buffer.insert_text('\n')

        @kb.add('enter', filter=~is_searching)
        def _(event):
            event.current_buffer.validate_and_handle()

        prompt_message = "user> "
        notifications = ""
        if self.new_files:
            self.new_files = False
            # prompt is a green asterix prepended to the prompt
            notifications = f"[green bold]*[/]{notifications}"
        if self.citations:
            notifications = f"[blue bold]*[/]{notifications}"

        prompt = str(rich.markup.render(f"{notifications}{prompt_message}"))

        files = [file for file in self.assistant.files]
        received_files = [file for file in self.received_files]
        files_completion = FileIdCompleter(files)
        received_completion = FileIdCompleter(received_files)

        completer = NestedCompleter.from_nested_dict({
            '!attach': files_completion,
            '!save': SaveFileCompleter(received_files),
            '!content': received_completion,
            '!citations': None,
            '!files': None,
            '!upload': PathCompleter(),
        })

        return self.session.prompt(
            prompt,
            key_bindings=kb,
            multiline=True,
            completer=completer,
            complete_while_typing=True
        ).strip()

    def add_citation(self, annotation) -> None:
        citation = annotation.file_citation.quote
        file_id = annotation.file_citation.file_id
        index = annotation.text

        file = self.client.files.retrieve(file_id)
        filename = file.filename

        self.citations[index] = (citation, filename, len(self.citations))

    def __parse_answer(self, message) -> None:
        content = message.content[0]
        if content.type == "text":
            text = message.content[0].text.value
            annotations = message.content[0].text.annotations
            self.citations = dict()
            system_messages = []
            for annotation in annotations:
                if annotation.type == "file_citation":
                    self.add_citation(annotation)
                    system_messages.append(self.CITATION_MESSAGE)

                elif annotation.type == "file_path":
                    fileid = annotation.file_path.file_id
                    try:
                        file = self.client.files.retrieve(fileid)
                    except openai.NotFoundError as e:
                        self.console.print(f"[bold red]{e.body}")
                        continue

                    system_messages.append(
                        f"Received file {file.filename} with id {fileid}\n"
                    )
                    self.new_files = True
                    self.received_files.append(file)

            syntax = Syntax(text, "markdown", theme="monokai", word_wrap=True)
            self.console.print("assistant> ")
            self.console.print(syntax)

            if system_messages:
                system_message = "\n".join(system_messages)
                self.console.print(f'[green bold]{system_message}')
        elif content.type == "image_file":
            file = self.client.files.retrieve(content.image_file.file_id)
            open(f"/tmp/{file.filename}", "wb").write(self.client.files.content(content.image_file.file_id).content)
            subprocess.Popen(["xdg-open", f"/tmp/{file.filename}"])

    def __display_steps(self, steps):
        for i, step in enumerate(steps):
            input, outputs = step
            if i not in self.processed_steps:
                self.processed_steps.append(i)
                syntax = Syntax(
                    input, "python",
                    theme="monokai",
                    line_numbers=True
                )
                panel = Panel(syntax)
                self.console.print("[bold yellow]Input:")
                self.console.print(panel)

            if len(outputs) and i not in self.outputed_steps:
                assert len(outputs) == 1
                if outputs[0].type == "logs":
                    self.console.print("[bold yellow]Output:")
                    panel = Panel(outputs[0].logs)
                    self.console.print(panel)
                    self.outputed_steps.append(i)
                if outputs[0].type == "image":
                    fileid = outputs[0].image.file_id
                    file = self.client.files.retrieve(fileid)
                    self.console.print(f"Received file image {file.filename} with id {fileid}")

    def __load_answer(self, run: GptRun):
        self.processed_steps = []
        self.outputed_steps = []

        try:
            with self.console.status("[bold green]Working on answer...") as status:
                while run.get_status() in ('queued', 'in_progress'):
                    if not (steps := run.collect_steps()):
                        continue
                    self.__display_steps(steps)
                if run.get_status() == "requires_action":
                    run.collect_steps()
                    self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=self.assistant.thread.id,
                        run_id=run.run.id,
                        tool_outputs=[
                            {
                                "tool_call_id": run.function[0],
                                "output": "Failed to send email, toto@maison.com does not exist."
                            }
                        ]
                    )
        except KeyboardInterrupt:
            self.client.beta.threads.runs.cancel(
                thread_id=run.thread_id,
                run_id=run.run.id
            )

            while run.get_status() not in ("cancelled", "completed"):
                time.sleep(0.1)

            self.console.print("[yellow]Request cancelled by the user")
            return None

        messages = self.assistant.messages()
        message = messages.data[0]

        return self.assistant.message(message.id)

    def cmd_package_search(self, python_module):
        packages = find_packages(python_module[0])
        for package in packages:
            print(f"{package['name']}: {package['description']}")

    def cmd_package_install(self, python_module):
        pass

    def cmd_multiline(self, _):
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        run = self.assistant.request("\n".join(lines))
        answer = self.__load_answer(run)
        if not answer:
            return

        self.__parse_answer(answer)

    def cmd_citations(self, _):
        for citation, filename, index in self.citations.values():
            self.console.print(f"[bold blue][{index}] {filename}:")
            self.console.print(f"[italic]{citation}")

    def cmd_upload(self, args):
        if not args:
            self.console.print("Usage: !upload <filepath>")
            return
        try:
            self.assistant.add_file(self.client.files.create(
                file=open(args[0], "rb"),
                purpose='assistants',
            ))
        except openai.APIError as e:
            # print in red
            self.console.print(f"[bold red]{e.body['message']}")

    def cmd_save(self, args):
        id, *dest = args[0].split(' ', maxsplit=1)

        try:
            file = self.client.files.retrieve(id)
            data = self.client.files.content(id).content
        except openai.APIError as e:
            self.console.print(f"[bold red]{e.body['message']}")
            return

        dest = dest[0] if dest else "/tmp/chatgpt/"
        if os.path.isdir(dest):
            folder = dest
            os.makedirs(folder, exist_ok=True)
            filename = os.path.basename(file.filename)
        else:
            folder = os.path.dirname(dest)
            os.makedirs(folder, exist_ok=True)
            filename = os.path.basename(dest)

        open(f"{folder}/{filename}", "wb").write(data)
        self.console.print(f"file saved in {folder}/{filename}")

    def cmd_attach(self, args):
        if not args:
            self.console.print("Usage: !attach <file_id>")
            return
        try:
            file = self.client.files.retrieve(args[0])
        except openai.NotFoundError as e:
            self.console.print(
                f"[bold red]{e.body.get('message', 'Unknown Error')}"
            )
            return

        self.assistant.add_file(file)

    def cmd_received_files(self, _):
        if not self.received_files:
            return

        max_filename_length = max(
            len(file.filename) for file in self.received_files
        )


        for file in self.received_files:
            filename = file.filename
            file_id = file.id

            created_epoch = file.created_at
            created_date = dt.utcfromtimestamp(created_epoch).strftime('%b %d %H:%M')

            file_size = file.bytes
            file_size_str = self.format_size(file_size)

            spacing = max_filename_length - len(filename) + 2

            desc = f"{filename: <{max_filename_length}} {file_size_str: >8} {created_date} [ {file_id} ]"
            print(desc)

    def format_size(self, size):
        if size < 1024:
            return f'{size}'
        elif size < 1024 ** 2:
            return f'{size / 1024:.2f}K'
        elif size < 1024 ** 3:
            return f'{size / (1024 ** 2):.2f}M'
        else:
            return f'{size / (1024 ** 3):.2f}G'

    def cmd_files(self, _):
        files = self.client.files.list().data
        if not files:
            return

        # Find the maximum length of the filenames
        max_filename_length = max(len(file.filename) for file in files)

        for file in files:
            filename = file.filename
            file_id = file.id

            # Calculate the spacing between the filename and file ID based on the filename length
            spacing = max_filename_length - len(filename) + 2

            desc = f"{filename: <{max_filename_length}} [ {file_id} ]"
            self.console.print(desc)

    def cmd_content(self, args):
        if not args:
            self.console.print("Usage: !content <file_id>")
            return
        try:
            file = self.client.files.retrieve(args[0])
        except openai.NotFoundError as e:
            self.console.print(f"[bold red]{e.body['message']}")
            return
        # extract extension:
        extension = os.path.splitext(file.filename)[1]
        try:
            text = self.client.files.content(args[0]).content.decode('utf-8')
        except openai.BadRequestError as e:
            self.console.print(f"[bold red]{e.body['message']}")
            return
        # Attempt to guess the lexer based on the code content
        match extension:
            case ".py":
                lexer = PythonLexer()
            case ".md":
                lexer = MarkdownLexer()
            case ".c":
                lexer = CLexer()
            case ".cpp" | ".cc":
                lexer = CppLexer()
            case _:
                lexer = guess_lexer(text)

        style_name = 'monokai'
        style = get_style_by_name(style_name)
        colorized_code = highlight(text, lexer, TerminalFormatter(style=style))

        print(colorized_code)

    def __internal_command(self, content):
        command, *args = content.split(maxsplit=1)
        command = command[1:]
        if internal_command := getattr(self, f"cmd_{command}", None):
            internal_command(args)
            return True
        else:
            self.console.print(f"Unknown command {command}")
            return False

    def process_prompt(self):
        content = self.__prompt()
        if not content:
            return

        if content.startswith("!"):
            self.__internal_command(content)
            return

        run = self.assistant.request(content)
        answer = self.__load_answer(run)
        if not answer:
            return

        self.__parse_answer(answer)

    def start(self):
        while True:
            try:
                self.process_prompt()
            except KeyboardInterrupt:
                continue
            except EOFError:
                break

        self.client.beta.threads.delete(self.assistant.thread.id)

chat = ChatApp()
chat.start()
