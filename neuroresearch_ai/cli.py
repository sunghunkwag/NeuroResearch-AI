"""
Command Line Interface for NeuroResearch-AI (with local mode)
"""

import asyncio
import typer
from pathlib import Path
from typing import Optional
import json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

from neuroresearch_ai import (
    NeuroResearchAI, 
    ResearchContext, 
    ResearchDomain, 
    ResearchTemplates
)

app = typer.Typer(
    name="neuroresearch",
    help="NeuroResearch-AI: Advanced Multi-Agent Research System",
    add_completion=False
)
console = Console()

@app.command()
def create_project(
    name: str = typer.Argument(..., help="Project name"),
    domain: str = typer.Option("ai_ml", help="Research domain"),
    project_dir: Optional[str] = typer.Option(None, help="Project directory"),
    template: Optional[str] = typer.Option(None, help="Research template to use")
):
    """Create a new research project"""
    
    if project_dir is None:
        project_dir = f"./{name}"
    
    # Map domain string to enum
    domain_map = {
        "ai_ml": ResearchDomain.AI_ML,
        "neuroscience": ResearchDomain.NEUROSCIENCE,
        "physics": ResearchDomain.PHYSICS,
        "chemistry": ResearchDomain.CHEMISTRY,
        "biology": ResearchDomain.BIOLOGY,
        "psychology": ResearchDomain.PSYCHOLOGY,
        "computer_science": ResearchDomain.COMPUTER_SCIENCE,
        "interdisciplinary": ResearchDomain.INTERDISCIPLINARY
    }
    
    domain_enum = domain_map.get(domain, ResearchDomain.AI_ML)
    
    # Create research context
    if template == "ai_research":
        context = ResearchTemplates.create_ai_research_template(name)
    elif template == "neuroscience":
        context = ResearchTemplates.create_neuroscience_template(name)
    else:
        context = ResearchContext(
            project_id=name,
            domain=domain_enum,
            research_question=""
        )
    
    # Create project directory structure
    project_path = Path(project_dir)
    project_path.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories
    for d in ("data","results","reports","configs"):
        (project_path / d).mkdir(exist_ok=True)
    
    # Save project configuration
    config_file = project_path / "configs" / "project_config.json"
    with open(config_file, 'w') as f:
        json.dump({
            "project_id": context.project_id,
            "domain": context.domain.value,
            "created_with": "NeuroResearch-AI v1.0.0"
        }, f, indent=2)
    
    console.print(Panel(
        f"Project '{name}' created successfully!\n"
        f"Location: {project_path}\n"
        f"Domain: {domain_enum.value}\n"
        f"Template: {template or 'default'}",
        title="NeuroResearch-AI Project Created",
        style="green"
    ))


@app.command()
def conduct_research(
    project_dir: str = typer.Argument(..., help="Project directory"),
    question: str = typer.Option(..., help="Research question"),
    config_file: Optional[str] = typer.Option(None, help="Configuration file"),
    quality_threshold: float = typer.Option(0.75, help="Quality threshold (0.0-1.0)"),
    max_iterations: int = typer.Option(5, help="Maximum iterations"),
    local: bool = typer.Option(False, help="Run in offline local mode (no API keys)"),
):
    """Conduct research using the AI agent system (cloud or local)"""
    
    async def run_research():
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            default_config = Path(project_dir) / "configs" / "project_config.json"
            if default_config.exists():
                with open(default_config, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
        
        # Prepare API keys only when not local
        if not local:
            api_keys = {
                "openai": typer.prompt("OpenAI API Key", hide_input=True),
            }
        else:
            api_keys = {}
        
        # Create research context
        context = ResearchContext(
            project_id=config.get("project_id", "research_project"),
            domain=ResearchDomain(config.get("domain", "ai_ml")),
            research_question=question
        )
        
        console.print("Initializing NeuroResearch-AI system...")
        research_system = NeuroResearchAI(
            project_dir=project_dir,
            research_context=context,
            api_keys=api_keys,
            enable_meta_learning=True,
            enable_cross_domain=True,
            enable_ethics_guardian=True,
            use_local_mode=local,
            quality_threshold=quality_threshold,
            max_iterations=max_iterations,
        )
        
        # Progress
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Conducting research...", total=None)
            results = await research_system.conduct_research(
                research_question=question,
            )
            progress.update(task, description="Research completed!")
        
        # Save report
        report = research_system.generate_research_report(results)
        report_file = Path(project_dir) / "reports" / f"research_report_{context.project_id}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        console.print(Panel(
            f"Research completed successfully!\n"
            f"Quality Score: {results.get('quality_score', 'N/A')}\n" 
            f"Report saved to: {report_file}",
            title="Research Results",
            style="green"
        ))
    
    asyncio.run(run_research())


@app.command()
def list_domains():
    """List available research domains"""
    
    table = Table(title="Available Research Domains")
    table.add_column("Domain", style="cyan")
    table.add_column("Description", style="green")
    
    domain_descriptions = {
        ResearchDomain.AI_ML: "Artificial Intelligence & Machine Learning",
        ResearchDomain.NEUROSCIENCE: "Neuroscience and Brain Research",
        ResearchDomain.PHYSICS: "Physics and Physical Sciences",
        ResearchDomain.CHEMISTRY: "Chemistry and Chemical Sciences",
        ResearchDomain.BIOLOGY: "Biology and Life Sciences",
        ResearchDomain.PSYCHOLOGY: "Psychology and Behavioral Sciences",
        ResearchDomain.COMPUTER_SCIENCE: "Computer Science and Software Engineering",
        ResearchDomain.INTERDISCIPLINARY: "Interdisciplinary Research"
    }
    
    for domain, description in domain_descriptions.items():
        table.add_row(domain.value, description)
    
    console.print(table)


@app.command()
def validate_project(
    project_dir: str = typer.Argument(..., help="Project directory to validate")
):
    """Validate project structure and configuration"""
    
    project_path = Path(project_dir)
    if not project_path.exists():
        console.print(f"Project directory does not exist: {project_dir}", style="red")
        raise typer.Exit(1)
    
    required_dirs = ["data", "results", "reports", "configs"]
    missing_dirs = [d for d in required_dirs if not (project_path / d).exists()]
    config_file = project_path / "configs" / "project_config.json"
    config_exists = config_file.exists()
    
    if missing_dirs or not config_exists:
        console.print("Project validation issues found:", style="yellow")
        if missing_dirs:
            console.print(f"Missing directories: {', '.join(missing_dirs)}")
        if not config_exists:
            console.print("Missing configuration file: configs/project_config.json")
    else:
        console.print("Project structure is valid!", style="green")


def main():
    app()


if __name__ == "__main__":
    main()
