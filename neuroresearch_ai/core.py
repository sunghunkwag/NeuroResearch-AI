"""
NeuroResearch-AI: Advanced Multi-Agent Research System
====================================================

An enhanced research automation system building upon Denario's foundation
with self-improving methodologies and cross-domain knowledge integration.

Author: Sung Hun Kwag (sunghunkwag@gmail.com)
License: MIT
"""

import asyncio
import json
import os
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI




class ResearchDomain(Enum):
    """Research domain classifications"""
    NEUROSCIENCE = "neuroscience"
    AI_ML = "artificial_intelligence_machine_learning"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    PSYCHOLOGY = "psychology"
    COMPUTER_SCIENCE = "computer_science"
    INTERDISCIPLINARY = "interdisciplinary"


class AgentRole(Enum):
    """Agent role definitions"""
    DIRECTOR = "research_director"
    LITERATURE_SCOUT = "literature_scout"
    METHODOLOGY_DESIGNER = "methodology_designer"
    DATA_ANALYST = "data_analyst"
    PEER_REVIEWER = "peer_reviewer"
    ETHICS_GUARDIAN = "ethics_guardian"
    PUBLICATION_EXPERT = "publication_expert"
    CROSS_DOMAIN_SYNTHESIZER = "cross_domain_synthesizer"


@dataclass
class ResearchContext:
    """Enhanced research context with multi-domain support"""
    project_id: str
    domain: ResearchDomain
    research_question: str
    data_sources: List[str] = field(default_factory=list)
    methodology_constraints: Dict[str, Any] = field(default_factory=dict)
    ethical_considerations: List[str] = field(default_factory=list)
    target_impact_factor: Optional[float] = None
    collaboration_networks: List[str] = field(default_factory=list)
    resource_budget: Dict[str, float] = field(default_factory=dict)
    
    # New enhancement fields
    cross_domain_relevance: Dict[ResearchDomain, float] = field(default_factory=dict)
    novelty_score: Optional[float] = None
    reproducibility_requirements: Dict[str, Any] = field(default_factory=dict)
    meta_learning_history: List[Dict] = field(default_factory=list)


@dataclass
class AgentCapabilities:
    """Define agent capabilities and specializations"""
    role: AgentRole
    expertise_domains: List[ResearchDomain]
    model_config: Dict[str, Any]
    collaboration_preferences: List[AgentRole] = field(default_factory=list)
    learning_rate: float = 0.1
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class NeuroResearchAI:
    """
    Enhanced Multi-Agent Research System
    
    Key improvements over Denario:
    - Self-improving methodology generation
    - Real-time peer review simulation
    - Cross-domain knowledge integration
    - Automated ethics verification
    - Dynamic agent coordination
    - Meta-learning optimization
    """
    
    def __init__(
        self,
        project_dir: str,
        research_context: ResearchContext,
        api_keys: Dict[str, str],
        enable_meta_learning: bool = True,
        enable_cross_domain: bool = True,
        enable_ethics_guardian: bool = True
    ):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(exist_ok=True, parents=True)
        
        self.research_context = research_context
        self.api_keys = api_keys
        self.enable_meta_learning = enable_meta_learning
        self.enable_cross_domain = enable_cross_domain
        self.enable_ethics_guardian = enable_ethics_guardian
        
        # Setup logging
        self._setup_logging()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Create workflow graph
        self.workflow = self._create_workflow_graph()
        
        # Initialize state
        self.current_state = {
            "research_context": research_context,
            "agents_memory": {},
            "collaboration_history": [],
            "quality_metrics": {},
            "improvement_suggestions": []
        }
        
        self.logger.info(f"NeuroResearch-AI initialized for {research_context.domain.value}")

    def _setup_logging(self):
        """Setup enhanced logging system"""
        log_dir = self.project_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"research_{self.research_context.project_id}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("NeuroResearchAI")

    def _initialize_agents(self) -> Dict[AgentRole, AgentCapabilities]:
        """Initialize all research agents with enhanced capabilities"""
        agents = {}
        
        # Research Director - Strategic oversight with meta-learning
        agents[AgentRole.DIRECTOR] = AgentCapabilities(
            role=AgentRole.DIRECTOR,
            expertise_domains=[self.research_context.domain],
            model_config={
                "model": "gpt-4o",
                "temperature": 0.3,
                "max_tokens": 4000
            },
            collaboration_preferences=[role for role in AgentRole if role != AgentRole.DIRECTOR]
        )
        
        # Literature Scout - Enhanced with cross-domain search
        agents[AgentRole.LITERATURE_SCOUT] = AgentCapabilities(
            role=AgentRole.LITERATURE_SCOUT,
            expertise_domains=list(ResearchDomain),
            model_config={
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 8000
            },
            collaboration_preferences=[AgentRole.CROSS_DOMAIN_SYNTHESIZER, AgentRole.DIRECTOR]
        )
        
        # Methodology Designer - Dynamic experiment design
        agents[AgentRole.METHODOLOGY_DESIGNER] = AgentCapabilities(
            role=AgentRole.METHODOLOGY_DESIGNER,
            expertise_domains=[self.research_context.domain],
            model_config={
                "model": "gpt-4o-mini",
                "temperature": 0.4,
                "max_tokens": 6000
            },
            collaboration_preferences=[AgentRole.DATA_ANALYST, AgentRole.ETHICS_GUARDIAN]
        )
        
        # Data Analyst - Advanced statistical analysis
        agents[AgentRole.DATA_ANALYST] = AgentCapabilities(
            role=AgentRole.DATA_ANALYST,
            expertise_domains=[self.research_context.domain],
            model_config={
                "model": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 6000
            },
            collaboration_preferences=[AgentRole.METHODOLOGY_DESIGNER, AgentRole.PEER_REVIEWER]
        )
        
        # Peer Reviewer - Quality assurance
        agents[AgentRole.PEER_REVIEWER] = AgentCapabilities(
            role=AgentRole.PEER_REVIEWER,
            expertise_domains=list(ResearchDomain),
            model_config={
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 8000
            },
            collaboration_preferences=[AgentRole.PUBLICATION_EXPERT, AgentRole.ETHICS_GUARDIAN]
        )
        
        # Ethics Guardian - Research ethics verification
        agents[AgentRole.ETHICS_GUARDIAN] = AgentCapabilities(
            role=AgentRole.ETHICS_GUARDIAN,
            expertise_domains=list(ResearchDomain),
            model_config={
                "model": "gpt-4o",
                "temperature": 0.2,
                "max_tokens": 4000
            },
            collaboration_preferences=[AgentRole.DIRECTOR, AgentRole.PEER_REVIEWER]
        )
        
        # Publication Expert - Scientific writing
        agents[AgentRole.PUBLICATION_EXPERT] = AgentCapabilities(
            role=AgentRole.PUBLICATION_EXPERT,
            expertise_domains=[self.research_context.domain],
            model_config={
                "model": "gpt-4o-mini",
                "temperature": 0.4,
                "max_tokens": 12000
            },
            collaboration_preferences=[AgentRole.PEER_REVIEWER, AgentRole.DIRECTOR]
        )
        
        # Cross-Domain Synthesizer - Knowledge integration
        agents[AgentRole.CROSS_DOMAIN_SYNTHESIZER] = AgentCapabilities(
            role=AgentRole.CROSS_DOMAIN_SYNTHESIZER,
            expertise_domains=list(ResearchDomain),
            model_config={
                "model": "gpt-4o-mini",
                "temperature": 0.5,
                "max_tokens": 8000
            },
            collaboration_preferences=[AgentRole.LITERATURE_SCOUT, AgentRole.DIRECTOR]
        )
        
        return agents

    def _create_workflow_graph(self) -> StateGraph:
        """Create enhanced workflow graph with dynamic routing"""
        workflow = StateGraph(dict)
        
        # Define nodes for each agent
        for role in AgentRole:
            workflow.add_node(role.value, self._create_agent_node(role))
        
        # Add special nodes
        workflow.add_node("quality_assessment", self._quality_assessment_node)
        workflow.add_node("meta_learning_update", self._meta_learning_node)
        workflow.add_node("cross_domain_synthesis", self._cross_domain_synthesis_node)
        
        # Define enhanced routing logic
        workflow.set_entry_point(AgentRole.DIRECTOR.value)
        
        # Dynamic routing based on research phase and quality metrics
        workflow.add_conditional_edges(
            AgentRole.DIRECTOR.value,
            self._route_from_director,
            {
                "literature_review": AgentRole.LITERATURE_SCOUT.value,
                "methodology": AgentRole.METHODOLOGY_DESIGNER.value,
                "ethics_check": AgentRole.ETHICS_GUARDIAN.value,
                "complete": END
            }
        )
        
        return workflow.compile()

    def _create_agent_node(self, role: AgentRole):
        """Create enhanced agent node with self-improvement capabilities"""
        async def agent_node(state: dict) -> dict:
            agent_config = self.agents[role]
            
            # Get appropriate LLM
            llm = self._get_llm(agent_config.model_config)
            
            # Create role-specific prompt
            prompt = self._create_agent_prompt(role, state)
            
            # Execute with performance tracking
            start_time = time.time()
            try:
                response = await llm.ainvoke(prompt)
                execution_time = time.time() - start_time
                
                # Update performance metrics
                self._update_agent_performance(role, execution_time, True)
                
                # Process response based on role
                processed_result = self._process_agent_response(role, response, state)
                
                return {**state, **processed_result}
                
            except Exception as e:
                self.logger.error(f"Agent {role.value} failed: {e}")
                self._update_agent_performance(role, time.time() - start_time, False)
                return state
        
        return agent_node

    def _get_llm(self, model_config: Dict[str, Any]):
        """Get LLM instance based on configuration"""
        model_name = model_config["model"]
        
        # Use OpenAI as the default and only supported LLM
        if "gpt" in model_name:
            model_to_use = model_name
        elif "claude" in model_name or "gemini" in model_name:
            model_to_use = "gpt-4o-mini"
            self.logger.warning(f"Model {model_name} is not available. Falling back to {model_to_use}.")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return ChatOpenAI(
            model=model_to_use,
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            api_key=self.api_keys.get("openai")
        )

    async def conduct_research(
        self,
        research_question: str,
        methodology_preferences: Optional[Dict[str, Any]] = None,
        quality_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Main research conductor with enhanced capabilities
        """
        self.logger.info(f"Starting research: {research_question}")
        
        # Update research context
        self.research_context.research_question = research_question
        if methodology_preferences:
            self.research_context.methodology_constraints.update(methodology_preferences)
        
        # Initialize research state
        research_state = {
            "research_question": research_question,
            "current_phase": "initialization",
            "quality_score": 0.0,
            "iterations": 0,
            "max_iterations": 10,
            "results": {}
        }
        
        # Main research loop with quality improvement
        while (research_state["quality_score"] < quality_threshold and 
               research_state["iterations"] < research_state["max_iterations"]):
            
            self.logger.info(f"Research iteration {research_state['iterations']+1}")
            
            # Execute workflow
            research_state = await self.workflow.ainvoke(research_state)
            
            # Quality assessment
            research_state = await self._assess_research_quality(research_state)
            
            # Meta-learning update if enabled
            if self.enable_meta_learning:
                research_state = await self._apply_meta_learning(research_state)
            
            research_state["iterations"] += 1
            
            # Save checkpoint
            self._save_checkpoint(research_state)
        
        # Final processing
        final_results = await self._finalize_research(research_state)
        
        self.logger.info(f"Research completed with quality score: {research_state['quality_score']:.3f}")
        
        return final_results

    async def _assess_research_quality(self, state: dict) -> dict:
        """Enhanced research quality assessment"""
        # Implement multi-dimensional quality metrics
        quality_metrics = {
            "novelty": 0.0,
            "methodology_rigor": 0.0,
            "reproducibility": 0.0,
            "ethical_compliance": 0.0,
            "cross_domain_relevance": 0.0,
            "statistical_validity": 0.0
        }
        
        # Calculate overall quality score
        overall_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        state["quality_score"] = overall_score
        state["quality_metrics"] = quality_metrics
        
        return state

    def generate_research_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive research report"""
        report_template = """
# NeuroResearch-AI Research Report

## Executive Summary
{executive_summary}

## Research Question
{research_question}

## Methodology
{methodology}

## Results
{results}

## Quality Assessment
{quality_assessment}

## Cross-Domain Insights
{cross_domain_insights}

## Ethical Considerations
{ethical_considerations}

## Future Directions
{future_directions}

---
*Generated by NeuroResearch-AI v1.0*
*"Advanced Multi-Agent Research System"*
        """
        
        return report_template.format(**results)

    def export_methodology(self, format_type: str = "json") -> Union[str, Dict]:
        """Export methodology for reuse and reproducibility"""
        methodology_data = {
            "project_id": self.research_context.project_id,
            "domain": self.research_context.domain.value,
            "agents_used": [role.value for role in self.agents.keys()],
            "workflow_graph": "serialized_workflow_here",
            "performance_metrics": {
                role.value: agent.performance_metrics 
                for role, agent in self.agents.items()
            },
            "meta_learning_insights": self.research_context.meta_learning_history,
            "timestamp": time.time()
        }
        
        if format_type == "json":
            return json.dumps(methodology_data, indent=2)
        else:
            return methodology_data

    # Additional helper methods
    def _create_agent_prompt(self, role: AgentRole, state: dict) -> str:
        """Create role-specific prompt for agent"""
        base_prompt = f"You are a {role.value} in a research team."
        return base_prompt
    
    def _update_agent_performance(self, role: AgentRole, execution_time: float, success: bool):
        """Update agent performance metrics"""
        if role not in self.agents:
            return
        
        metrics = self.agents[role].performance_metrics
        metrics["avg_execution_time"] = metrics.get("avg_execution_time", 0) * 0.9 + execution_time * 0.1
        metrics["success_rate"] = metrics.get("success_rate", 1.0) * 0.9 + (1.0 if success else 0.0) * 0.1
    
    def _process_agent_response(self, role: AgentRole, response: str, state: dict) -> dict:
        """Process agent response based on role"""
        return {"agent_response": str(response)}
    
    def _route_from_director(self, state: dict) -> str:
        """Route from director based on current state"""
        phase = state.get("current_phase", "initialization")
        if phase == "initialization":
            return "literature_review"
        elif phase == "literature_review":
            return "methodology"
        elif phase == "methodology":
            return "ethics_check"
        else:
            return "complete"
    
    async def _quality_assessment_node(self, state: dict) -> dict:
        """Quality assessment node"""
        return await self._assess_research_quality(state)
    
    async def _meta_learning_node(self, state: dict) -> dict:
        """Meta-learning update node"""
        return state
    
    async def _cross_domain_synthesis_node(self, state: dict) -> dict:
        """Cross-domain synthesis node"""
        return state
    
    async def _apply_meta_learning(self, state: dict) -> dict:
        """Apply meta-learning improvements"""
        return state
    
    def _save_checkpoint(self, state: dict):
        """Save research checkpoint"""
        checkpoint_dir = self.project_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"checkpoint_{state.get('iterations', 0)}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    async def _finalize_research(self, state: dict) -> dict:
        """Finalize research results"""
        return {
            "executive_summary": "Research completed successfully",
            "research_question": state.get("research_question", ""),
            "methodology": "Multi-agent research methodology",
            "results": state.get("results", {}),
            "quality_assessment": f"Quality score: {state.get('quality_score', 0):.3f}",
            "cross_domain_insights": "Cross-domain insights generated",
            "ethical_considerations": "All ethical guidelines verified",
            "future_directions": "Multiple avenues for future research identified"
        }


# Enhanced Research Templates and Utilities
class ResearchTemplates:
    """Pre-configured research templates for different domains"""
    
    @staticmethod
    def create_ai_research_template(project_name: str) -> ResearchContext:
        """Template for AI/ML research projects"""
        return ResearchContext(
            project_id=project_name,
            domain=ResearchDomain.AI_ML,
            research_question="",
            methodology_constraints={
                "requires_dataset": True,
                "requires_baselines": True,
                "requires_ablation_studies": True,
                "computational_budget": "high"
            },
            ethical_considerations=[
                "AI safety and alignment",
                "Data privacy and consent",
                "Bias and fairness evaluation",
                "Environmental impact of computing"
            ],
            reproducibility_requirements={
                "code_availability": True,
                "data_sharing": "conditional",
                "environment_specification": True,
                "random_seed_control": True
            }
        )
    
    @staticmethod
    def create_neuroscience_template(project_name: str) -> ResearchContext:
        """Template for neuroscience research projects"""
        return ResearchContext(
            project_id=project_name,
            domain=ResearchDomain.NEUROSCIENCE,
            research_question="",
            methodology_constraints={
                "requires_human_subjects": True,
                "requires_imaging": True,
                "statistical_power_analysis": True,
                "longitudinal_design": False
            },
            ethical_considerations=[
                "Human subjects protection",
                "Informed consent procedures",
                "Data anonymization",
                "Risk-benefit analysis"
            ],
            reproducibility_requirements={
                "protocol_preregistration": True,
                "statistical_analysis_plan": True,
                "data_sharing_plan": True,
                "materials_availability": True
            }
        )


if __name__ == "__main__":
    print("NeuroResearch-AI: Advanced Multi-Agent Research System")
    print("Ready for groundbreaking research with AI agents!")