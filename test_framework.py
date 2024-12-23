import unittest
import asyncio
import torch
from autonomous_learner import AutonomousLearner
from search_manager import BraveSearchManager
from pattern_recognition import PatternRecognizer

class TestAutonomousLearner(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Enable CUDA memory caching
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Pre-allocate CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            # Warm up GPU
            dummy = torch.ones(1).cuda()
            del dummy
            
    async def asyncSetUp(self):
        if not hasattr(TestAutonomousLearner, '_learner'):
            TestAutonomousLearner._learner = AutonomousLearner()
            TestAutonomousLearner._search_manager = BraveSearchManager()
            TestAutonomousLearner._pattern_recognizer = PatternRecognizer()
        
        self.learner = TestAutonomousLearner._learner
        self.search_manager = TestAutonomousLearner._search_manager
        self.pattern_recognizer = TestAutonomousLearner._pattern_recognizer
        
    async def test_search_functionality(self):
        query = "artificial intelligence recent developments"
        results = await self.search_manager.search(query, count=3)
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results, list)
        
    async def test_pattern_recognition(self):
        texts = [
            "AI has made significant progress",
            "Artificial Intelligence shows major advances",
            "Climate change affects global weather",
            "Global warming impacts climate patterns"
        ]
        patterns = self.pattern_recognizer.find_patterns(texts)
        self.assertIsInstance(patterns, dict)
        
    async def test_knowledge_integration(self):
        info = {
            'content': 'Test information',
            'confidence': 0.8
        }
        self.learner.integrate_knowledge(info)
        knowledge = self.learner.get_current_knowledge()
        self.assertIsInstance(knowledge, list)
        
    async def test_threat_assessment(self):
        input_data = {
            'content': 'Safe test content',
            'source': 'trusted_source.com'
        }
        threat_level = self.learner.deep_system.assess_threat(input_data)
        self.assertIsInstance(threat_level, float)
        self.assertLess(threat_level, 1.0)

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()

if __name__ == '__main__':
    asyncio.run(unittest.main())
