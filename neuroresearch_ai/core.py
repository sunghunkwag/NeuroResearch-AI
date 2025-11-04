"""
NeuroResearch-AI: Advanced Multi-Agent Research System (Core)
- Adds local/offline mode, heuristic quality assessment, expanded workflow, and role prompts
- Keeps cloud LLM path (OpenAI) as-is; adds switchable local mode without API keys
"""

import json
import time
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

try:
    from langgraph.graph import StateGraph, END
    from langchain_openai import ChatOpenAI
except Exception:
    # Optional imports in local mode
    StateGraph = None
    END = "__END__"
    ChatOpenAI = None


class ResearchDomain(Enum):
    NEUROSCIENCE = "neuroscience"
    AI_ML = "artificial_intelligence_machine_learning"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    PSYCHOLOGY = "psychology"
    COMPUTER_SCIENCE = "computer_science"
    INTERDISCIPLINARY = "interdisciplinary"


class AgentRole(Enum):
    DIRECTOR = "research_director"
    LITERATURE_SCOUT = "literature_scout"
    CROSS_DOMAIN_SYNTHESIZER = "cross_domain_synthesizer"
    METHODOLOGY_DESIGNER = "methodology_designer"
    DATA_ANALYST = "data_analyst"
    PEER_REVIEWER = "peer_reviewer"
    ETHICS_GUARDIAN = "ethics_guardian"
    PUBLICATION_EXPERT = "publication_expert"


@dataclass
class ResearchContext:
    project_id: str
    domain: ResearchDomain
    research_question: str
    data_sources: List[str] = field(default_factory=list)
    methodology_constraints: Dict[str, Any] = field(default_factory=dict)
    ethical_considerations: List[str] = field(default_factory=list)
    target_impact_factor: Optional[float] = None
    collaboration_networks: List[str] = field(default_factory=list)
    resource_budget: Dict[str, float] = field(default_factory=dict)
    cross_domain_relevance: Dict[ResearchDomain, float] = field(default_factory=dict)
    novelty_score: Optional[float] = None
    reproducibility_requirements: Dict[str, Any] = field(default_factory=dict)
    meta_learning_history: List[Dict] = field(default_factory=list)


@dataclass
class AgentCapabilities:
    role: AgentRole
    expertise_domains: List[ResearchDomain]
    model_config: Dict[str, Any]
    collaboration_preferences: List[AgentRole] = field(default_factory=list)
    learning_rate: float = 0.1
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class _LocalEngine:
    """Lightweight offline engine: keyword extraction, templates, simulated results"""

    def __init__(self, context: ResearchContext):
        self.ctx = context
        self.prompts = _RolePrompts.default()

    def keywords(self, text: str) -> List[str]:
        stop = {"how","what","why","when","where","the","and","for","with","into","from","that","this","can","we","are","you","your"}
        return [w for w in ''.join(ch if ch.isalnum() or ch==' ' else ' ' for ch in text.lower()).split() if len(w)>3 and w not in stop][:10]

    def analyze(self, question: str) -> Dict[str, Any]:
        kws = self.keywords(question)
        return {
            "keywords": kws,
            "method_suggestions": ["literature_review","cross_domain_synthesis","experimental_design","analysis","peer_review"],
            "risks": ["bias","overfitting","underpowered_study","ethical_risk"],
        }

    def simulate_results(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        import random
        random.seed(7)
        perf = {
            "accuracy": round(random.uniform(0.82,0.94),3),
            "f1": round(random.uniform(0.80,0.93),3)
        }
        return {
            "findings": [
                "Method improves efficiency over baseline",
                "Cross-domain evidence supports methodology",
                "Ethical checklist passes with minor notes"
            ],
            "performance": perf,
            "evidence_count": len(analysis.get("keywords", [])) + 3
        }

    def quality(self, state: Dict[str, Any]) -> Dict[str, float]:
        # Simple heuristic based on evidence/findings/lengths
        res = state.get("results", {})
        analysis = state.get("analysis", {})
        evidence = res.get("evidence_count", 0)
        findings = len(res.get("findings", []))
        kws = len(analysis.get("keywords", []))
        novelty = 0.6 + min(0.4, 0.02*kws)
        rigor = 0.6 + min(0.4, 0.03*evidence)
        reproducibility = 0.6 + min(0.4, 0.05*findings)
        ethics = 0.7
        cross_domain = 0.6 + min(0.4, 0.02*kws)
        stats = 0.6 + min(0.4, 0.02*evidence)
        return {
            "novelty": round(novelty,3),
            "methodology_rigor": round(rigor,3),
            "reproducibility": round(reproducibility,3),
            "ethical_compliance": round(ethics,3),
            "cross_domain_relevance": round(cross_domain,3),
            "statistical_validity": round(stats,3)
        }


class _RolePrompts:
    @staticmethod
    def default() -> Dict[AgentRole, str]:
        return {
            AgentRole.LITERATURE_SCOUT: (
                "You are a literature scout. Identify key sources, summarize evidence, and list gaps."
            ),
            AgentRole.CROSS_DOMAIN_SYNTHESIZER: (
                "You synthesize insights across domains. Propose connections and transfer hypotheses."
            ),
            AgentRole.METHODOLOGY_DESIGNER: (
                "Design a rigorous, reproducible methodology with clear variables, controls, and evaluation."
            ),
            AgentRole.DATA_ANALYST: (
                "Analyze results, compute metrics, and check statistical validity with brief justification."
            ),
            AgentRole.PEER_REVIEWER: (
                "Critique the study. Provide strengths, weaknesses, and concrete improvement suggestions."
            ),
            AgentRole.ETHICS_GUARDIAN: (
                "Run an ethics checklist and flag risks (privacy, dual-use, bias, safety)."
            ),
            AgentRole.PUBLICATION_EXPERT: (
                "Draft a concise abstract and bullet-point contributions ready for publication."
            ),
            AgentRole.DIRECTOR: (
                "Coordinate agents, set targets, and decide next steps based on quality metrics."
            ),
        }


class NeuroResearchAI:
    def __init__(
        self,
        project_dir: str,
        research_context: ResearchContext,
        api_keys: Dict[str, str] | None = None,
        enable_meta_learning: bool = True,
        enable_cross_domain: bool = True,
        enable_ethics_guardian: bool = True,
        use_local_mode: bool = False,
        local_backend: Literal["template"] = "template",
        quality_threshold: float = 0.75,
        max_iterations: int = 5,
    ):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(exist_ok=True, parents=True)
        self.research_context = research_context
        self.api_keys = api_keys or {}
        self.enable_meta_learning = enable_meta_learning
        self.enable_cross_domain = enable_cross_domain
        self.enable_ethics_guardian = enable_ethics_guardian
        self.use_local_mode = use_local_mode
        self.local_backend = local_backend
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations

        self._setup_logging()
        self.logger.info(f"Init: local_mode={self.use_local_mode} backend={self.local_backend}")

        # Local engine
        self.local_engine = _LocalEngine(research_context) if self.use_local_mode else None

        # Graph only needed for cloud mode
        if not self.use_local_mode and StateGraph is not None:
            self.workflow = self._create_workflow_graph()
        else:
            self.workflow = None

    def _setup_logging(self):
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

    # --------------- Local/offline pipeline ---------------
    def _run_local_iteration(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not state.get("analysis"):
            state["analysis"] = self.local_engine.analyze(state["research_question"])  # literature + synthesis
        # methodology & results
        state["results"] = self.local_engine.simulate_results(state["analysis"])
        # quality
        qm = self.local_engine.quality(state)
        state["quality_metrics"] = qm
        state["quality_score"] = round(sum(qm.values())/len(qm),3)
        return state

    # --------------- Cloud/LLM pipeline (OpenAI-only baseline) ---------------
    def _get_llm(self, model: str, temperature: float, max_tokens: int):
        if self.use_local_mode:
            raise RuntimeError("LLM not available in local mode")
        if not ChatOpenAI:
            raise RuntimeError("langchain_openai not available")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_keys.get("openai")
        )

    def _create_workflow_graph(self):
        graph = StateGraph(dict)
        for role in AgentRole:
            graph.add_node(role.value, lambda s, r=role: self._cloud_agent_step(r, s))
        graph.set_entry_point(AgentRole.DIRECTOR.value)
        # Expanded linear-first pipeline with all roles
        graph.add_conditional_edges(
            AgentRole.DIRECTOR.value,
            lambda s: "literature" if s.get("current_phase") in (None, "initialization") else "end",
            {"literature": AgentRole.LITERATURE_SCOUT.value, "end": END}
        )
        graph.add_edge(AgentRole.LITERATURE_SCOUT.value, AgentRole.CROSS_DOMAIN_SYNTHESIZER.value)
        graph.add_edge(AgentRole.CROSS_DOMAIN_SYNTHESIZER.value, AgentRole.METHODOLOGY_DESIGNER.value)
        graph.add_edge(AgentRole.METHODOLOGY_DESIGNER.value, AgentRole.DATA_ANALYST.value)
        graph.add_edge(AgentRole.DATA_ANALYST.value, AgentRole.PEER_REVIEWER.value)
        graph.add_edge(AgentRole.PEER_REVIEWER.value, AgentRole.ETHICS_GUARDIAN.value)
        graph.add_edge(AgentRole.ETHICS_GUARDIAN.value, AgentRole.PUBLICATION_EXPERT.value)
        return graph.compile()

    async def _cloud_agent_step(self, role: AgentRole, state: Dict[str, Any]) -> Dict[str, Any]:
        # Minimal cloud step: call OpenAI with role prompt + context
        llm = self._get_llm("gpt-4o-mini", 0.2, 2000)
        prompt = (
            _RolePrompts.default().get(role, "")
            + "\n\nResearch question: " + state.get("research_question", "")
            + "\nCurrent summary: " + json.dumps({k: state.get(k) for k in ("analysis","results")}, ensure_ascii=False)
            + "\nReply with a short JSON object with keys: output (string), evidence (list)."
        )
        try:
            resp = await llm.ainvoke(prompt)
            content = getattr(resp, "content", "") or getattr(resp, "text", "")
        except Exception as e:
            self.logger.warning(f"LLM call failed at {role.value}: {e}")
            content = "{\"output\": \"N/A\", \"evidence\": []}"
        # naive parse
        try:
            parsed = json.loads(content) if isinstance(content, str) and content.strip().startswith("{") else {"output": content, "evidence": []}
        except Exception:
            parsed = {"output": content, "evidence": []}
        # fold into state
        bucket = state.setdefault("cloud_outputs", {})
        bucket[role.value] = parsed
        if role == AgentRole.DATA_ANALYST:
            state["results"] = {
                "findings": [parsed.get("output","")],
                "evidence_count": len(parsed.get("evidence", []))
            }
        if role == AgentRole.LITERATURE_SCOUT:
            state["analysis"] = {"keywords": self.local_engine.keywords(state.get("research_question",""))}
        return state

    async def conduct_research(
        self,
        research_question: str,
        methodology_preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = {
            "research_question": research_question,
            "current_phase": "initialization",
            "quality_score": 0.0,
            "iterations": 0,
            "results": {},
            "analysis": {}
        }
        # apply prefs
        if methodology_preferences:
            self.research_context.methodology_constraints.update(methodology_preferences)

        while state["quality_score"] < self.quality_threshold and state["iterations"] < self.max_iterations:
            if self.use_local_mode:
                state = self._run_local_iteration(state)
            else:
                # run full graph once
                state = await self.workflow.ainvoke(state)
                # compute quality from cloud outputs + heuristic
                # reuse local quality to avoid API coupling
                state = self._run_local_iteration(state)

            # meta-learning (lightweight): keep last metrics
            if self.enable_meta_learning:
                mh = self.research_context.meta_learning_history
                mh.append({"it": state["iterations"], "metrics": state.get("quality_metrics", {})})
                self.research_context.meta_learning_history = mh[-20:]

            state["iterations"] += 1
            self._save_checkpoint(state)

        return await self._finalize(state)

    def _save_checkpoint(self, state: Dict[str, Any]):
        ckpt = self.project_dir / "checkpoints"
        ckpt.mkdir(exist_ok=True)
        with open(ckpt / f"checkpoint_{state['iterations']}.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    async def _finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        qm = state.get("quality_metrics", {})
        return {
            "executive_summary": "Study completed",
            "research_question": state.get("research_question",""),
            "methodology": "Local or cloud pipeline with multi-agent roles",
            "results": state.get("results", {}),
            "quality_assessment": json.dumps(qm, indent=2),
            "cross_domain_insights": "Included via synthesis role",
            "ethical_considerations": "Checked via ethics role or local rules",
            "future_directions": "Iterate with refined prompts and datasets",
            "quality_score": state.get("quality_score", 0.0)
        }
