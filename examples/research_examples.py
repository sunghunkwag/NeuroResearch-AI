"""
Example usage of NeuroResearch-AI for AI/ML research
"""

import asyncio
from neuroresearch_ai import (
    NeuroResearchAI,
    ResearchContext,
    ResearchDomain,
    ResearchTemplates
)


async def ai_research_example():
    """Example: AI research on transformer attention mechanisms"""
    
    # Create research context using template
    context = ResearchTemplates.create_ai_research_template("transformer_attention_research")
    context.research_question = "How can we improve transformer attention mechanisms for processing extremely long sequences while maintaining computational efficiency?"
    
    # Add specific constraints for this research
    context.methodology_constraints.update({
        "sequence_lengths": [1024, 4096, 16384, 65536],
        "datasets": ["arxiv", "books", "code"],
        "baseline_models": ["standard_transformer", "longformer", "performer"],
        "evaluation_metrics": ["perplexity", "memory_usage", "training_time", "inference_speed"]
    })
    
    # API keys (in practice, load from environment or config)
    api_keys = {
        "openai": "your-openai-key-here",
        "anthropic": "your-anthropic-key-here",
        "google": "your-google-key-here"
    }
    
    # Initialize the research system
    research_system = NeuroResearchAI(
        project_dir="./ai_research_project",
        research_context=context,
        api_keys=api_keys,
        enable_meta_learning=True,
        enable_cross_domain=True,
        enable_ethics_guardian=True
    )
    
    print("Starting AI research on transformer attention mechanisms...")
    
    # Conduct research
    results = await research_system.conduct_research(
        research_question=context.research_question,
        methodology_preferences={
            "experimental_validation": True,
            "theoretical_analysis": True,
            "ablation_studies": True,
            "comparative_evaluation": True
        },
        quality_threshold=0.85
    )
    
    print(f"Research completed with quality score: {results.get('quality_score', 'N/A'):.3f}")
    
    # Generate comprehensive report
    report = research_system.generate_research_report(results)
    
    # Save report
    with open("./ai_research_project/reports/transformer_attention_report.md", "w") as f:
        f.write(report)
    
    print("Research report saved!")
    
    # Export methodology for future use
    methodology = research_system.export_methodology("json")
    with open("./ai_research_project/configs/methodology_export.json", "w") as f:
        f.write(methodology)
    
    print("Methodology exported for reproducibility!")
    
    return results


async def neuroscience_research_example():
    """Example: Neuroscience research on memory consolidation"""
    
    # Create neuroscience research context
    context = ResearchTemplates.create_neuroscience_template("memory_consolidation_study")
    context.research_question = "What are the neural mechanisms underlying memory consolidation during sleep in different age groups?"
    
    # Add neuroscience-specific constraints
    context.methodology_constraints.update({
        "participant_groups": ["young_adults", "middle_aged", "elderly"],
        "sample_size_per_group": 30,
        "neuroimaging_methods": ["fMRI", "EEG"],
        "sleep_stages": ["NREM1", "NREM2", "NREM3", "REM"],
        "memory_tasks": ["declarative", "procedural", "working_memory"]
    })
    
    # Enhanced ethical considerations for human subjects research
    context.ethical_considerations.extend([
        "IRB approval required",
        "Informed consent for neuroimaging",
        "Data anonymization protocols",
        "Participant screening for contraindications",
        "Right to withdraw without penalty"
    ])
    
    api_keys = {
        "openai": "your-openai-key-here",
        "anthropic": "your-anthropic-key-here",
        "google": "your-google-key-here"
    }
    
    research_system = NeuroResearchAI(
        project_dir="./neuroscience_project",
        research_context=context,
        api_keys=api_keys,
        enable_meta_learning=True,
        enable_cross_domain=True,
        enable_ethics_guardian=True  # Critical for human subjects research
    )
    
    print("Starting neuroscience research on memory consolidation...")
    
    results = await research_system.conduct_research(
        research_question=context.research_question,
        methodology_preferences={
            "randomized_controlled_design": True,
            "power_analysis": True,
            "multi_modal_neuroimaging": True,
            "longitudinal_followup": True
        },
        quality_threshold=0.9  # Higher threshold for human subjects research
    )
    
    print(f"Neuroscience research completed with quality score: {results.get('quality_score', 'N/A'):.3f}")
    
    # Generate report with emphasis on ethical compliance
    report = research_system.generate_research_report(results)
    
    with open("./neuroscience_project/reports/memory_consolidation_report.md", "w") as f:
        f.write(report)
    
    print("Neuroscience research report saved!")
    
    return results


async def interdisciplinary_research_example():
    """Example: Interdisciplinary research combining AI and neuroscience"""
    
    # Create custom interdisciplinary research context
    context = ResearchContext(
        project_id="brain_inspired_ai_architectures",
        domain=ResearchDomain.INTERDISCIPLINARY,
        research_question="Can we develop more efficient AI architectures by incorporating principles from neuroscience research on attention and memory?",
        methodology_constraints={
            "requires_neuroscience_literature_review": True,
            "requires_ai_implementation": True,
            "requires_comparative_evaluation": True,
            "computational_resources": "high",
            "interdisciplinary_collaboration": True
        },
        ethical_considerations=[
            "Responsible AI development",
            "Proper attribution of neuroscience insights",
            "Consideration of dual-use implications",
            "Environmental impact of large-scale experiments"
        ],
        cross_domain_relevance={
            ResearchDomain.NEUROSCIENCE: 0.8,
            ResearchDomain.AI_ML: 0.9,
            ResearchDomain.PSYCHOLOGY: 0.6,
            ResearchDomain.COMPUTER_SCIENCE: 0.7
        }
    )
    
    api_keys = {
        "openai": "your-openai-key-here", 
        "anthropic": "your-anthropic-key-here",
        "google": "your-google-key-here"
    }
    
    research_system = NeuroResearchAI(
        project_dir="./interdisciplinary_project",
        research_context=context,
        api_keys=api_keys,
        enable_meta_learning=True,
        enable_cross_domain=True,  # Critical for interdisciplinary research
        enable_ethics_guardian=True
    )
    
    print("Starting interdisciplinary AI-neuroscience research...")
    
    results = await research_system.conduct_research(
        research_question=context.research_question,
        methodology_preferences={
            "cross_domain_literature_synthesis": True,
            "bio_inspired_architecture_design": True,
            "empirical_validation": True,
            "theoretical_framework_development": True,
            "interdisciplinary_peer_review": True
        },
        quality_threshold=0.88
    )
    
    print(f"Interdisciplinary research completed with quality score: {results.get('quality_score', 'N/A'):.3f}")
    
    report = research_system.generate_research_report(results)
    
    with open("./interdisciplinary_project/reports/brain_inspired_ai_report.md", "w") as f:
        f.write(report)
    
    print("Interdisciplinary research report saved!")
    
    return results


async def main():
    """Run all examples"""
    
    print("NeuroResearch-AI Examples")
    print("=" * 50)
    
    # Run AI research example
    print("\n1. AI/ML Research Example:")
    try:
        await ai_research_example()
        print("AI research example completed successfully!")
    except Exception as e:
        print(f"AI research example failed: {e}")
    
    print("\n" + "-" * 50)
    
    # Run neuroscience research example  
    print("\n2. Neuroscience Research Example:")
    try:
        await neuroscience_research_example()
        print("Neuroscience research example completed successfully!")
    except Exception as e:
        print(f"Neuroscience research example failed: {e}")
    
    print("\n" + "-" * 50)
    
    # Run interdisciplinary research example
    print("\n3. Interdisciplinary Research Example:")
    try:
        await interdisciplinary_research_example()
        print("Interdisciplinary research example completed successfully!")
    except Exception as e:
        print(f"Interdisciplinary research example failed: {e}")
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    # Note: In practice, you would set your API keys as environment variables
    print("Remember to set your API keys before running these examples!")
    print("Set OPENAI_API_KEY, ANTHROPIC_API_KEY, and GOOGLE_API_KEY environment variables")
    print("\nTo run examples:")
    print("python examples/research_examples.py")
    
    # Uncomment to run examples:
    # asyncio.run(main())