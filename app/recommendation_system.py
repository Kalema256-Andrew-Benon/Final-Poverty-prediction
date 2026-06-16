"""
==========================================
POVERTY CLASS RECOMMENDATION SYSTEM
==========================================
Production-ready module for Poverty Streamlit Application.
Handles retrieval, validation, formatting, and delivery of 
poverty alleviation recommendations based on predicted class.
"""

import json
import os
from typing import Dict, List, Optional

# Database path relative to this module
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recommendations.json")


def _load_database() -> Dict:
    """Safely load the recommendations JSON database."""
    if not os.path.exists(_DB_PATH):
        return {}
    try:
        with open(_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def validate_class_id(class_id: int) -> bool:
    """Check if a class ID exists in the database."""
    db = _load_database()
    return str(class_id) in db


def get_recommendations(class_id: int, count: int = 10) -> Dict:
    """
    Retrieve tailored recommendations for a predicted poverty class.
    
    Args:
        class_id: Predicted class (0=Extreme, 1=Moderate, 2=Low)
        count: Number of recommendations to return (default: 10)
        
    Returns:
        Dictionary with label, focus_areas, recommendations list, and metadata.
    """
    db = _load_database()
    str_id = str(class_id)
    
    if str_id not in db:
        return {
            'class_id': class_id,
            'label': 'Unknown Class',
            'focus_areas': ['General Support'],
            'recommendations': [
                'Please consult a local social services office for personalized guidance.',
                'Verify the prediction class and try again.'
            ],
            'returned_count': 2,
            'is_valid': False
        }
    
    data = db[str_id]
    recs = data.get('recommendations', [])
    return {
        'class_id': class_id,
        'label': data.get('label', 'Unknown'),
        'focus_areas': data.get('focus_areas', []),
        'recommendations': recs[:count],
        'total_available': len(recs),
        'returned_count': min(count, len(recs)),
        'is_valid': True
    }


def format_for_ui(result: Dict, numbered: bool = True) -> str:
    """
    Format recommendation result for Streamlit display.
    
    Args:
        result: Output dictionary from get_recommendations()
        numbered: Whether to prefix recommendations with numbers
        
    Returns:
        Formatted string ready for st.markdown()
    """
    lines = [f"### 🎯 {result['label']}"]
    if result['focus_areas']:
        lines.append(f"**Focus Areas:** {', '.join(result['focus_areas'])}")
    lines.append("")
    lines.append("#### 💡 Recommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        prefix = f"{i}. " if numbered else "• "
        lines.append(f"{prefix}{rec}")
    return "\n".join(lines)


def get_available_classes() -> List[Dict]:
    """Return list of all supported classes with metadata."""
    db = _load_database()
    return [
        {'id': int(k), 'label': v.get('label', 'Unknown'), 'count': len(v.get('recommendations', []))}
        for k, v in sorted(db.items(), key=lambda x: int(x[0]))
    ]
