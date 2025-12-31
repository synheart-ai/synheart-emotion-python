"""Basic usage example for synheart-emotion package."""
from datetime import datetime

from synheart_emotion import EmotionConfig, EmotionEngine


def main():
    """Run basic emotion inference example."""
    # Create engine with 60s window, 5s step configuration (for faster demo)
    config = EmotionConfig(
        window_seconds=60.0,  # 60 second window
        step_seconds=5.0,  # 5 second step
    )
    engine = EmotionEngine.from_pretrained(config)

    # Simulate wearable data over time
    # In a real application, this would come from a wearable device
    from datetime import timedelta
    
    # Generate data points with timestamps spread over 59 seconds
    # This fills the 60s window: from 59s ago to "now" (ensures oldest is >= 58s required)
    now = datetime.now()
    num_points = 13  # 13 points spaced ~4.9s apart = 59 seconds total
    
    for i in range(num_points):
        hr = 72.0 + (i % 3) * 1.5  # Varying HR
        rr_intervals = [
            850.0 + (i % 5) * 10.0 + (j * 2.0)
            for j in range(8)  # 8 RR intervals per sample
        ]
        
        # Create timestamps: evenly spaced from 59s ago to now
        # i=0: 59s ago, i=12: now (0s ago)
        # This ensures oldest point is 59s old (>= 58s required) and within 60s window
        if i == num_points - 1:
            timestamp = now  # Last point is exactly "now"
        else:
            seconds_ago = 59 - (i * (59.0 / (num_points - 1)))
            timestamp = now - timedelta(seconds=seconds_ago)
        
        engine.push(
            hr=hr,
            rr_intervals_ms=rr_intervals,
            timestamp=timestamp,
        )
        print(f"Pushed data point {i + 1}: HR={hr:.1f}, RR count={len(rr_intervals)}")

    # Get inference results when ready
    results = engine.consume_ready()
    if results:
        for result in results:
            print(f"\n=== Emotion Inference Result ===")
            print(f"Emotion: {result.emotion}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Probabilities:")
            for emotion, prob in result.probabilities.items():
                print(f"  {emotion}: {prob:.1%}")
            print(f"Features: {result.features}")
    else:
        print("\nNo results ready yet (need more data or time)")

    # Get buffer statistics
    stats = engine.get_buffer_stats()
    print(f"\n=== Buffer Statistics ===")
    print(f"Data points: {stats['count']}")
    print(f"Duration: {stats['duration_ms']}ms")
    print(f"HR range: {stats['hr_range']}")
    print(f"Total RR intervals: {stats['rr_count']}")


if __name__ == "__main__":
    main()
