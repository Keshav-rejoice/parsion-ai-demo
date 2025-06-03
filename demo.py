from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env

from praisonaiagents import Agent, Task, PraisonAIAgents

# Define the Researcher agent
researcher = Agent(
    name="Researcher",
    role="AI Researcher",
    goal="Investigate cutting-edge AI trends",
    backstory="Expert data scientist with a passion for AI.",
    llm="gpt-4o",     # You can change to "gpt-4", "gpt-3.5-turbo", etc.
    verbose=True
)

# Define the Writer agent
writer = Agent(
    name="Writer",
    role="Content Strategist",
    goal="Write an engaging article based on research",
    backstory="Skilled writer with a flair for making tech accessible.",
    llm="gpt-4o",
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the latest developments in AI and data science",
    expected_output="A detailed report on recent AI breakthroughs",
    agent=researcher
)

write_task = Task(
    description="Write a blog post based on the AI research findings",
    expected_output="A well-structured blog post on AI developments",
    agent=writer,
    context=[research_task]  # This connects tasks (dependency)
)

# Run agents
agents = PraisonAIAgents(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True
)

# Execute the workflow
result = agents.start()
print("\nFinal Output:\n", result)
