def get_opened_recommendations(user_feedback):
    opened_recommendations = [
        event
        for event in user_feedback
        if event["event_name"] == "notification_opened"
        and event["properties"]["content_type"] == "recommendation"
    ]
    return opened_recommendations


def get_rated_recommendations(user_feedback):
    rated_recommendations = [
        event
        for event in user_feedback
        if event["event_name"] == "notification_rated"
        and event["properties"]["content_type"] == "recommendation"
    ]
    return rated_recommendations


def get_rated_resources(user_feedback):
    rated_resources = [
        event
        for event in user_feedback
        if event["event_name"] == "notification_rated"
        and event["properties"]["content_type"] == "resource"
    ]
    return rated_resources
