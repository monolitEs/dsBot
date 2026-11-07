import discord

class RoleManager:

    def __init__(self, bot):
        self.bot = bot

    async def create_role(self, message, role, r, g, b):

        role_color = discord.Color.from_rgb(r, g, b)

        new_role = await message.guild.create_role(name=role, permissions=discord.Permissions.all(), color=role_color)

        await message.author.add_roles(new_role)

