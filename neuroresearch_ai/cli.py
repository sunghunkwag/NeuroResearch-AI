"""
Command Line Interface for NeuroResearch-AI (Enhanced)
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
    help="NeuroResearch-AI: Advanced Multi-Agent Research System with local/cloud modes",
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
    for d in ("data", "results", "reports", "configs", "templates", "checkpoints"):
        (project_path / d).mkdir(exist_ok=True)
    
    # Save project configuration
    config_file = project_path / "configs" / "project_config.json"
    with open(config_file, 'w') as f:
        json.dump({
            "project_id": context.project_id,
            "domain": context.domain.value,
            "created_with": "NeuroResearch-AI v1.0.0",
            "supports_local_mode": True,
            "supports_autogen": True,
            "supports_multi_llm": True
        }, f, indent=2)
    
    console.print(Panel(
        f"Project '{name}' created successfully!\n"
        f"Location: {project_path}\n"
        f"Domain: {domain_enum.value}\n"
        f"Template: {template or 'default'}\n\n"
        f"Features enabled:\n"
        f"• Local/offline mode support\n"
        f"• Multi-LLM provider support\n"
        f"• AutoGen integration\n"
        f"• Meta-learning optimization",
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
    use_autogen: bool = typer.Option(False, help="Enable AutoGen agent conversations"),
    provider: str = typer.Option("openai", help="LLM provider: openai, anthropic, google, or mixed"),
    model: Optional[str] = typer.Option(None, help="Specific model to use (overrides provider defaults)"),
):
    """Conduct research using the AI agent system (local/cloud/hybrid modes)"""
    
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
        api_keys = {}
        if not local:
            console.print(f"\nConfiguring {provider} API access...", style="blue")
            
            if provider in ("openai", "mixed"):
                openai_key = typer.prompt("OpenAI API Key", hide_input=True)
                api_keys["openai"] = openai_key
            
            if provider in ("anthropic", "mixed"):
                anthropic_key = typer.prompt("Anthropic API Key (optional)", default="", hide_input=True)
                if anthropic_key:
                    api_keys["anthropic"] = anthropic_key
            
            if provider in ("google", "mixed"):
                google_key = typer.prompt("Google API Key (optional)", default="", hide_input=True)
                if google_key:
                    api_keys["google"] = google_key
        
        # Create research context
        context = ResearchContext(
            project_id=config.get("project_id", "research_project"),
            domain=ResearchDomain(config.get("domain", "ai_ml")),
            research_question=question
        )
        
        # Display configuration
        mode_info = [
            f"Mode: {'Local/Offline' if local else 'Cloud'}",
            f"AutoGen: {'Enabled' if use_autogen else 'Disabled'}",
            f"Provider: {provider if not local else 'N/A (Local)'}",
            f"Quality Threshold: {quality_threshold}",
            f"Max Iterations: {max_iterations}"
        ]
        
        console.print(Panel(
            "\n".join(mode_info),
            title="Research Configuration",
            style="blue"
        ))
        
        console.print("Initializing NeuroResearch-AI system...")
        research_system = NeuroResearchAI(
            project_dir=project_dir,
            research_context=context,
            api_keys=api_keys,
            enable_meta_learning=True,
            enable_cross_domain=True,
            enable_ethics_guardian=True,
            use_local_mode=local,
            use_autogen=use_autogen,
            quality_threshold=quality_threshold,
            max_iterations=max_iterations,
        )
        
        # Progress tracking
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
        
        # Save methodology export
        methodology = research_system.export_methodology("json")
        methodology_file = Path(project_dir) / "configs" / "methodology_export.json"
        with open(methodology_file, 'w', encoding='utf-8') as f:
            f.write(methodology)
        
        # Display results
        result_info = [
            f"Quality Score: {results.get('quality_score', 'N/A'):.3f}",
            f"Mode Used: {results.get('mode', 'unknown')}",
            f"AutoGen: {results.get('autogen_enabled', False)}",
            f"Report: {report_file}",
            f"Methodology: {methodology_file}"
        ]
        
        console.print(Panel(
            "\n".join(result_info),
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
def list_providers():
    """List available LLM providers and models"""
    
    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Models", style="green")
    table.add_column("API Key Required", style="yellow")
    table.add_column("Local Mode", style="blue")
    
    providers_info = [
        ("OpenAI", "gpt-4o, gpt-4o-mini, gpt-3.5-turbo", "Yes", "No"),
        ("Anthropic", "claude-3-opus, claude-3-sonnet, claude-3-haiku", "Yes", "No"),
        ("Google", "gemini-1.5-pro, gemini-1.0-pro", "Yes", "No"),
        ("Local", "Template-based heuristics", "No", "Yes"),
        ("Mixed", "All providers (failover)", "At least one", "Partial")
    ]
    
    for provider, models, api_key, local in providers_info:
        table.add_row(provider, models, api_key, local)
    
    console.print(table)
    
    console.print("\n[bold]Usage Examples:[/bold]")
    console.print("• Local mode (no API keys): --local")
    console.print("• OpenAI only: --provider openai")
    console.print("• Mixed providers: --provider mixed")
    console.print("• With AutoGen: --use-autogen")


@app.command()
def validate_project(
    project_dir: str = typer.Argument(..., help="Project directory to validate")
):
    """Validate project structure and configuration"""
    
    project_path = Path(project_dir)
    if not project_path.exists():
        console.print(f"Project directory does not exist: {project_dir}", style="red")
        raise typer.Exit(1)
    
    required_dirs = ["data", "results", "reports", "configs", "templates", "checkpoints"]
    missing_dirs = [d for d in required_dirs if not (project_path / d).exists()]
    
    config_file = project_path / "configs" / "project_config.json"
    config_exists = config_file.exists()
    
    # Check templates
    templates_dir = project_path / "templates" / "prompts"
    templates_exist = templates_dir.exists()
    
    if missing_dirs or not config_exists:
        console.print("Project validation issues found:", style="yellow")
        if missing_dirs:
            console.print(f"Missing directories: {', '.join(missing_dirs)}")
        if not config_exists:
            console.print("Missing configuration file: configs/project_config.json")
        if not templates_exist:
            console.print("Templates directory will be created automatically")
    else:
        console.print("Project structure is valid!", style="green")
        
        # Show project info
        if config_exists:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            info_table = Table(title="Project Information")
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="green")
            
            for key, value in config.items():
                info_table.add_row(str(key), str(value))
            
            console.print(info_table)


@app.command()
def test_config(
    project_dir: str = typer.Argument(..., help="Project directory"),
    provider: str = typer.Option("local", help="Provider to test: local, openai, anthropic, google")
):
    """Test configuration and API connectivity"""
    
    async def run_test():
        console.print(f"Testing {provider} configuration...", style="blue")
        
        # Create minimal test context
        context = ResearchContext(
            project_id="config_test",
            domain=ResearchDomain.AI_ML,
            research_question="Test configuration"
        )
        
        api_keys = {}
        if provider != "local":
            if provider == "openai":
                api_keys["openai"] = typer.prompt("OpenAI API Key", hide_input=True)
            elif provider == "anthropic":
                api_keys["anthropic"] = typer.prompt("Anthropic API Key", hide_input=True)
            elif provider == "google":
                api_keys["google"] = typer.prompt("Google API Key", hide_input=True)
        
        try:
            system = NeuroResearchAI(
                project_dir=project_dir,
                research_context=context,
                api_keys=api_keys,
                use_local_mode=(provider == "local"),
                max_iterations=1,
                quality_threshold=0.1
            )
            
            console.print(f"✓ System initialized successfully", style="green")
            
            # Quick test run
            console.print("Running quick test...", style="blue")
            results = await system.conduct_research("Test question for configuration")
            
            console.print(f"✓ Test completed successfully", style="green")
            console.print(f"Quality score: {results.get('quality_score', 'N/A')}", style="cyan")
            console.print(f"Mode: {results.get('mode', 'unknown')}", style="cyan")
            
        except Exception as e:
            console.print(f"✗ Test failed: {e}", style="red")
            raise typer.Exit(1)
    
    asyncio.run(run_test())


def main():
    app()


if __name__ == "__main__":
    main()