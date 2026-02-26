from fastmcp import FastMCP

mcp = FastMCP("Basic tools")


@mcp.tool()
def flight_search(origin: str, destination: str, date: str) -> str:
    """Search flights between two cities for a given date."""
    return f"Flights from {origin} to {destination} on {date}"


@mcp.tool()
def hotel_search(city: str, check_in: str, nights: int) -> str:
    """Find hotels in a city for specific dates."""
    return f"Hotels in {city} from {check_in} for {nights} nights"


@mcp.tool()
def restaurant_finder(city: str, cuisine: str) -> str:
    """Find restaurants by cuisine in a city."""
    return f"{cuisine} restaurants in {city}"


@mcp.tool()
def weather_lookup(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather for {city}: sunny"


@mcp.tool()
def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency amounts."""
    return f"{amount} {from_currency} in {to_currency}"


@mcp.tool()
def translate_text(text: str, target_language: str) -> str:
    """Translate text to a target language."""
    return f"Translated to {target_language}: {text}"


@mcp.tool()
def book_movie_tickets(movie: str, city: str, time: str) -> str:
    """Book movie tickets for a showtime."""
    return f"Booked {movie} in {city} at {time}"


@mcp.tool()
def book_ride(pickup: str, dropoff: str) -> str:
    """Book a ride between two locations."""
    return f"Ride booked from {pickup} to {dropoff}"


@mcp.tool()
def schedule_meeting(title: str, day: str, time: str) -> str:
    """Schedule a meeting on a calendar."""
    return f"Meeting '{title}' scheduled on {day} at {time}"


@mcp.tool()
def summarize_article(url: str) -> str:
    """Summarize the content of a web article."""
    return f"Summary for {url}"


@mcp.tool()
def find_flights_deals(city: str) -> str:
    """Find cheap flight deals from a city."""
    return f"Deals from {city}"


@mcp.tool()
def book_saloon(city: str, service: str) -> str:
    """Book a salon appointment."""
    return f"Booked {service} in {city}"


@mcp.tool()
def order_food(city: str, cuisine: str) -> str:
    """Order food delivery by cuisine."""
    return f"Ordered {cuisine} in {city}"


@mcp.tool()
def get_news(topic: str) -> str:
    """Fetch the latest news about a topic."""
    return f"News about {topic}"


@mcp.tool()
def plan_trip(city: str, days: int) -> str:
    """Generate a simple trip plan."""
    return f"Trip plan for {city} ({days} days)"


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8001)
