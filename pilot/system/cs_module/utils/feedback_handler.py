def get_sent_notifications(user_feedback):
    """Return all notification_sent events (recommendation or resource)."""
    return [e for e in user_feedback if e["event_name"] == "notification_sent"]


def get_sent_recommendations(user_feedback):
    sent_recommendations = [
        event
        for event in user_feedback
        if event["event_name"] == "notification_sent" and event["properties"]["content_type"] == "recommendation"
    ]
    return sent_recommendations


def get_opened_recommendations(user_feedback):
    opened_recommendations = [
        event
        for event in user_feedback
        if event["event_name"] == "notification_opened" and event["properties"]["content_type"] == "recommendation"
    ]
    return opened_recommendations


def get_rated_recommendations(user_feedback):
    rated_recommendations = [
        event
        for event in user_feedback
        if event["event_name"] == "notification_rated" and event["properties"]["content_type"] == "recommendation"
    ]
    return rated_recommendations


def get_rated_resources(user_feedback):
    rated_resources = [
        event
        for event in user_feedback
        if event["event_name"] == "notification_rated" and event["properties"]["content_type"] == "resource"
    ]
    return rated_resources
