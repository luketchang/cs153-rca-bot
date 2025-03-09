import asyncio
import logging
import os
from pathlib import Path

import discord
from discord.ext import commands
from dotenv import load_dotenv

from oncall.agent.agentv3 import OncallAgent
from oncall.agent.nodes.utils import get_formatted_labels_map, load_file_tree
from oncall.chat.models import SupportTicket
from oncall.chat.response_generator import ResponseGenerator
from oncall.constants import DISCORD_TOKEN
from oncall.lib.utils import get_llm

PREFIX = "!"

logger = logging.getLogger("discord")

load_dotenv()

# Discord setup
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Agents setup
reasoning_llm = get_llm("o3-mini-2025-01-31")
fast_llm = get_llm("gpt-4o-2024-08-06")
generator = ResponseGenerator(fast_llm)
agent = OncallAgent(reasoning_llm, fast_llm)

# Repo setup/info
labels_map = get_formatted_labels_map("2025-03-03 03:00:00", "2025-03-03 03:30:00")
repo_path = Path("/Users/luketchang/code/Microservices_Udemy/ticketing")
overview_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "codebase_overviews",
    "typescript-microservices.txt",
)
with open(overview_path, "r") as f:
    overview = f.read()

file_tree = load_file_tree(repo_path)


# Discord event handlers
@bot.event
async def on_ready():
    logger.info(f"{bot.user} has connected to Discord!")


@bot.event
async def on_message(message: discord.Message):
    await bot.process_commands(message)

    if message.author.bot or message.content.startswith("!"):
        return

    logger.info(f"Processing message from {message.author}: {message.content}")

    N = 10
    message_history = []
    async for msg in message.channel.history(limit=N):
        if not msg.author.bot:
            message_history.append(msg.content)
    message_history.reverse()

    loop = asyncio.get_event_loop()
    response_obj = await loop.run_in_executor(
        None, lambda: generator.generate_response(message_history)
    )

    if isinstance(response_obj.ticket_or_followup, str):
        reply_message = response_obj.ticket_or_followup
    elif isinstance(response_obj.ticket_or_followup, SupportTicket):
        ticket = response_obj.ticket_or_followup
        formatted_ticket = f"Support Ticket:\nDescription: {ticket.description}\nTime: {ticket.datetime}"

        state = {
            "first_pass": True,
            "issue": formatted_ticket,
            "code_request": "Search code in the payments and orders directories specifically, nothing else.",
            "log_request": "Search logs from the payments and orders services specifically.",
            "request": "Search for any code in the payments service related to the issue.",
            "repo_path": str(repo_path),
            "codebase_overview": overview,
            "file_tree": file_tree,
            "labels_map": labels_map,
            "chat_history": [],
        }

        # Inform the user the bot is working on a long-running task.
        await message.reply(
            "Thank you for letting us know. We will start investigating the issue right now..."
        )
        async with message.channel.typing():
            response = await asyncio.to_thread(agent.invoke, state)
        reply_message = response["rca"]

    await message.reply(reply_message)


@bot.command(name="ping", help="Pings the bot.")
async def ping(ctx, *, arg=None):
    if arg is None:
        await ctx.send("Pong!")
    else:
        await ctx.send(f"Pong! Your argument was {arg}")


bot.run(DISCORD_TOKEN)
