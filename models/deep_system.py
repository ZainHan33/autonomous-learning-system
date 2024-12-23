import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

class DeepSystem(nn.Module):
    """Deep system for unconscious processing"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.processor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(0.2)
        )
        self.threat_detector = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        self.core_values = {
            'stability': 1.0,
            'learning_rate': 0.01,
            'novelty_threshold': 0.3,
            'integrity_threshold': 0.8,
            'adaptation_rate': 0.1
        }
        self.threat_memory = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        features = self.processor(x)
        threat = self.threat_detector(features)
        return features, threat
    
    def assess_threat(self, input_data: Dict) -> float:
        """Assess threat level of input data"""
        try:
            # Convert input data to feature vector
            features = self._extract_features(input_data)
            
            # Process through threat detector
            with torch.no_grad():
                _, threat = self.forward(features)
            
            base_threat = float(threat.cpu().item())
            
            # Add contextual threat assessment
            context_threat = self._assess_contextual_threat(input_data)
            
            # Combine threats with weights
            total_threat = 0.7 * base_threat + 0.3 * context_threat
            
            # Store in threat memory if significant
            if total_threat > 0.5:
                self.threat_memory.append({
                    'data': input_data,
                    'threat_level': total_threat,
                    'timestamp': datetime.now()
                })
                
            return total_threat
            
        except Exception as e:
            print(f"Threat assessment error: {str(e)}")
            return 1.0  # Conservative approach - treat as threat if assessment fails
    
    def _extract_features(self, input_data: Dict) -> torch.Tensor:
        """Extract feature vector from input data"""
        features = torch.zeros(100).to(self.device)
        
        try:
            # Extract text features if available
            if 'content' in input_data:
                content = input_data['content']
                # Use basic text stats as features
                features[0] = len(content) / 1000  # Length
                features[1] = len(content.split()) / 100  # Word count
                features[2] = len(set(content.split())) / 100  # Unique words
                
            # Source reliability features
            if 'source' in input_data:
                source = input_data['source']
                features[3] = 1.0 if 'https' in source else 0.0  # HTTPS
                features[4] = 1.0 if '.edu' in source or '.gov' in source else 0.0  # Trustworthy domain
                
            # Add timestamp features
            if 'timestamp' in input_data:
                timestamp = input_data['timestamp']
                features[5] = (datetime.now() - timestamp).days / 30  # Age in months
                
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            
        return features.unsqueeze(0)  # Add batch dimension
    
    def _assess_contextual_threat(self, input_data: Dict) -> float:
        """Assess threat based on context and core values"""
        threat_level = 0.0
        
        # Check stability impact
        if len(self.threat_memory) > 0:
            recent_threats = [t['threat_level'] for t in self.threat_memory[-10:]]
            stability_threat = np.mean(recent_threats) if recent_threats else 0.0
            threat_level += 0.3 * stability_threat
        
        # Check value alignment
        for value, threshold in self.core_values.items():
            if value in input_data:
                deviation = abs(input_data[value] - threshold) / threshold
                threat_level += 0.2 * deviation
        
        return min(1.0, threat_level)