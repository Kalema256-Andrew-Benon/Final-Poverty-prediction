
import json
import os

def get_recommendations(class_id: int, count: int = 10) -> dict:
    """
    Retrieves recommendations based on the predicted poverty class.
    Reads from recommendations.json in the app directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'recommendations.json')
    
    try:
        with open(json_path, 'r') as f:
            db = json.load(f)
    except FileNotFoundError:
        return {
            'label': 'Error',
            'recommendations': ['Recommendation database not found.'],
            'focus_areas': [],
            'returned_count': 0
        }

    str_id = str(class_id)
    
    if str_id not in db:
        return {
            'label': 'Unknown Class',
            'recommendations': ['No specific recommendations found for this class.'],
            'focus_areas': [],
            'returned_count': 0
        }
    
    data = db[str_id]
    recs = data.get('recommendations', [])
    
    return {
        'label': data.get('label', 'Unknown'),
        'focus_areas': data.get('focus_areas', []),
        'recommendations': recs[:count],
        'returned_count': min(count, len(recs))
    }
