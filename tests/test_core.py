"""
Test cases for NeuroResearch-AI core functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile
import shutil

from neuroresearch_ai import (
    NeuroResearchAI,
    ResearchContext,
    ResearchDomain,
    AgentRole,
    ResearchTemplates
)


class TestResearchContext:
    """Test ResearchContext class"""
    
    def test_research_context_creation(self):
        """Test basic research context creation"""
        context = ResearchContext(
            project_id="test_project",
            domain=ResearchDomain.AI_ML,
            research_question="Test question"
        )
        
        assert context.project_id == "test_project"
        assert context.domain == ResearchDomain.AI_ML
        assert context.research_question == "Test question"
        assert len(context.data_sources) == 0
        assert len(context.ethical_considerations) == 0
    
    def test_research_context_with_constraints(self):
        """Test research context with methodology constraints"""
        context = ResearchContext(
            project_id="test_project",
            domain=ResearchDomain.NEUROSCIENCE,
            research_question="Test question",
            methodology_constraints={
                "requires_human_subjects": True,
                "sample_size": 100
            },
            ethical_considerations=[
                "IRB approval required",
                "Informed consent"
            ]
        )
        
        assert context.methodology_constraints["requires_human_subjects"] is True
        assert context.methodology_constraints["sample_size"] == 100
        assert len(context.ethical_considerations) == 2


class TestResearchTemplates:
    """Test research template generation"""
    
    def test_ai_research_template(self):
        """Test AI research template creation"""
        template = ResearchTemplates.create_ai_research_template("ai_project")
        
        assert template.project_id == "ai_project"
        assert template.domain == ResearchDomain.AI_ML
        assert template.methodology_constraints["requires_dataset"] is True
        assert "AI safety and alignment" in template.ethical_considerations
    
    def test_neuroscience_template(self):
        """Test neuroscience research template creation"""
        template = ResearchTemplates.create_neuroscience_template("neuro_project")
        
        assert template.project_id == "neuro_project"
        assert template.domain == ResearchDomain.NEUROSCIENCE
        assert template.methodology_constraints["requires_human_subjects"] is True
        assert "Human subjects protection" in template.ethical_considerations


class TestNeuroResearchAI:
    """Test main NeuroResearchAI class"""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_api_keys(self):
        """Mock API keys for testing"""
        return {
            "openai": "test-openai-key",
            "anthropic": "test-anthropic-key", 
            "google": "test-google-key"
        }
    
    @pytest.fixture
    def test_context(self):
        """Test research context"""
        return ResearchContext(
            project_id="test_research",
            domain=ResearchDomain.AI_ML,
            research_question="How can we test AI research systems?"
        )
    
    def test_neuroresearch_ai_initialization(self, temp_project_dir, test_context, mock_api_keys):
        """Test NeuroResearchAI initialization"""
        with patch('neuroresearch_ai.core.logging'):
            research_system = NeuroResearchAI(
                project_dir=temp_project_dir,
                research_context=test_context,
                api_keys=mock_api_keys
            )
            
            assert research_system.project_dir == Path(temp_project_dir)
            assert research_system.research_context == test_context
            assert research_system.api_keys == mock_api_keys
            assert research_system.enable_meta_learning is True
            assert research_system.enable_cross_domain is True
            assert research_system.enable_ethics_guardian is True
    
    def test_agent_initialization(self, temp_project_dir, test_context, mock_api_keys):
        """Test agent initialization"""
        with patch('neuroresearch_ai.core.logging'):
            research_system = NeuroResearchAI(
                project_dir=temp_project_dir,
                research_context=test_context,
                api_keys=mock_api_keys
            )
            
            # Check that all required agents are initialized
            expected_agents = [
                AgentRole.DIRECTOR,
                AgentRole.LITERATURE_SCOUT,
                AgentRole.METHODOLOGY_DESIGNER,
                AgentRole.DATA_ANALYST,
                AgentRole.PEER_REVIEWER,
                AgentRole.ETHICS_GUARDIAN,
                AgentRole.PUBLICATION_EXPERT,
                AgentRole.CROSS_DOMAIN_SYNTHESIZER
            ]
            
            for agent_role in expected_agents:
                assert agent_role in research_system.agents
                agent = research_system.agents[agent_role]
                assert agent.role == agent_role
                assert "model" in agent.model_config
                assert "temperature" in agent.model_config
                assert "max_tokens" in agent.model_config
    
    @pytest.mark.asyncio
    async def test_conduct_research_mock(self, temp_project_dir, test_context, mock_api_keys):
        """Test research conduct with mocked LLM calls"""
        with patch('neuroresearch_ai.core.logging'):
            research_system = NeuroResearchAI(
                project_dir=temp_project_dir,
                research_context=test_context,
                api_keys=mock_api_keys
            )
            
            # Mock the workflow execution
            mock_results = {
                "research_question": "Test question",
                "quality_score": 0.85,
                "results": {"analysis": "Test analysis complete"},
                "iterations": 3
            }
            
            with patch.object(research_system, 'workflow') as mock_workflow:
                mock_workflow.ainvoke = AsyncMock(return_value=mock_results)
                
                with patch.object(research_system, '_assess_research_quality') as mock_assess:
                    mock_assess.return_value = mock_results
                    
                    with patch.object(research_system, '_finalize_research') as mock_finalize:
                        mock_finalize.return_value = mock_results
                        
                        with patch.object(research_system, '_save_checkpoint'):
                            results = await research_system.conduct_research(
                                research_question="Test question",
                                quality_threshold=0.8
                            )
                            
                            assert results["quality_score"] == 0.85
                            assert "research_question" in results
    
    def test_llm_selection(self, temp_project_dir, test_context, mock_api_keys):
        """Test LLM model selection"""
        with patch('neuroresearch_ai.core.logging'):
            research_system = NeuroResearchAI(
                project_dir=temp_project_dir,
                research_context=test_context,
                api_keys=mock_api_keys
            )
            
            # Test OpenAI model selection
            with patch('neuroresearch_ai.core.ChatOpenAI') as mock_openai:
                model_config = {"model": "gpt-4o", "temperature": 0.3, "max_tokens": 4000}
                research_system._get_llm(model_config)
                mock_openai.assert_called_once()
            
            # Test Anthropic model selection
            with patch('neuroresearch_ai.core.ChatAnthropic') as mock_anthropic:
                model_config = {"model": "claude-3-sonnet-20240229", "temperature": 0.2, "max_tokens": 8000}
                research_system._get_llm(model_config)
                mock_anthropic.assert_called_once()
            
            # Test Google model selection
            with patch('neuroresearch_ai.core.ChatGoogleGenerativeAI') as mock_google:
                model_config = {"model": "gemini-1.5-pro", "temperature": 0.4, "max_tokens": 6000}
                research_system._get_llm(model_config)
                mock_google.assert_called_once()
    
    def test_report_generation(self, temp_project_dir, test_context, mock_api_keys):
        """Test research report generation"""
        with patch('neuroresearch_ai.core.logging'):
            research_system = NeuroResearchAI(
                project_dir=temp_project_dir,
                research_context=test_context,
                api_keys=mock_api_keys
            )
            
            test_results = {
                "executive_summary": "Test summary",
                "research_question": "Test question",
                "methodology": "Test methodology",
                "results": "Test results",
                "quality_assessment": "High quality",
                "cross_domain_insights": "Interdisciplinary connections",
                "ethical_considerations": "All ethical requirements met",
                "future_directions": "Promising areas for future work"
            }
            
            report = research_system.generate_research_report(test_results)
            
            assert "NeuroResearch-AI Research Report" in report
            assert "Test summary" in report
            assert "Test question" in report
            assert "Test methodology" in report
    
    def test_methodology_export(self, temp_project_dir, test_context, mock_api_keys):
        """Test methodology export functionality"""
        with patch('neuroresearch_ai.core.logging'):
            research_system = NeuroResearchAI(
                project_dir=temp_project_dir,
                research_context=test_context,
                api_keys=mock_api_keys
            )
            
            # Test JSON export
            methodology_json = research_system.export_methodology("json")
            assert isinstance(methodology_json, str)
            assert "project_id" in methodology_json
            assert "test_research" in methodology_json
            
            # Test dict export
            methodology_dict = research_system.export_methodology("dict")
            assert isinstance(methodology_dict, dict)
            assert methodology_dict["project_id"] == "test_research"
            assert methodology_dict["domain"] == "artificial_intelligence_machine_learning"


@pytest.mark.asyncio
async def test_integration_example():
    """Integration test with minimal setup"""
    with tempfile.TemporaryDirectory() as temp_dir:
        context = ResearchTemplates.create_ai_research_template("integration_test")
        context.research_question = "How can we test AI research integration?"
        
        api_keys = {
            "openai": "test-key",
            "anthropic": "test-key",
            "google": "test-key"
        }
        
        with patch('neuroresearch_ai.core.logging'):
            research_system = NeuroResearchAI(
                project_dir=temp_dir,
                research_context=context,
                api_keys=api_keys
            )
            
            # Verify initialization worked
            assert research_system.research_context.project_id == "integration_test"
            assert len(research_system.agents) == 8  # All 8 agents should be initialized
            
            print("✅ Integration test passed!")


if __name__ == "__main__":
    pytest.main([__file__])