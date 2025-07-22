"""Calendar and time-related functions plugin."""

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import calendar
from .function_registry import get_function_registry
from .function_types import FunctionDefinition, FunctionType, FunctionParameter, ParameterType

def get_current_time(format: str = "%Y-%m-%d %H:%M:%S", timezone_name: str = "UTC") -> Dict[str, Any]:
    """
    Get current date and time in various formats.
    
    Args:
        format: Time format string or special format name
        timezone_name: Timezone name
        
    Returns:
        Dictionary containing formatted time and metadata
    """
    try:
        # Get current time
        now = datetime.now(timezone.utc)
        
        # Handle special format names
        if format == "iso":
            formatted_time = now.isoformat()
        elif format == "timestamp":
            formatted_time = int(now.timestamp())
        elif format == "unix":
            formatted_time = int(now.timestamp())
        else:
            formatted_time = now.strftime(format)
        
        return {
            'time': formatted_time,
            'format': format,
            'timezone': timezone_name,
            'timestamp': int(now.timestamp()),
            'iso': now.isoformat(),
            'weekday': now.strftime("%A"),
            'month': now.strftime("%B")
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'time': None
        }

def add_time(base_time: str, days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0) -> Dict[str, Any]:
    """
    Add time to a base datetime.
    
    Args:
        base_time: Base time in ISO format or timestamp
        days: Days to add
        hours: Hours to add
        minutes: Minutes to add
        seconds: Seconds to add
        
    Returns:
        Dictionary containing the new time
    """
    try:
        # Parse base time
        if base_time.isdigit():
            base_dt = datetime.fromtimestamp(int(base_time), timezone.utc)
        else:
            base_dt = datetime.fromisoformat(base_time.replace('Z', '+00:00'))
        
        # Add time delta
        delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        new_time = base_dt + delta
        
        return {
            'original_time': base_time,
            'new_time': new_time.isoformat(),
            'added': {
                'days': days,
                'hours': hours,
                'minutes': minutes,
                'seconds': seconds
            },
            'timestamp': int(new_time.timestamp())
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'new_time': None
        }

def get_business_days(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Calculate business days between two dates.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Dictionary containing business day calculation
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        business_days = 0
        current = start
        
        while current <= end:
            if current.weekday() < 5:  # Monday = 0, Sunday = 6
                business_days += 1
            current += timedelta(days=1)
        
        total_days = (end - start).days + 1
        weekend_days = total_days - business_days
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'business_days': business_days,
            'total_days': total_days,
            'weekend_days': weekend_days
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'business_days': None
        }

def get_calendar_info(year: int, month: int) -> Dict[str, Any]:
    """
    Get calendar information for a specific month.
    
    Args:
        year: Year (e.g., 2024)
        month: Month (1-12)
        
    Returns:
        Dictionary containing calendar information
    """
    try:
        # Get calendar for the month
        cal = calendar.monthcalendar(year, month)
        
        # Get month name
        month_name = calendar.month_name[month]
        
        # Count weekdays and weekends
        weekdays = 0
        weekends = 0
        
        for week in cal:
            for day in week:
                if day != 0:  # 0 represents days from other months
                    date = datetime(year, month, day)
                    if date.weekday() < 5:
                        weekdays += 1
                    else:
                        weekends += 1
        
        # Get first and last day of month
        first_day = datetime(year, month, 1)
        last_day = datetime(year, month, calendar.monthrange(year, month)[1])
        
        return {
            'year': year,
            'month': month,
            'month_name': month_name,
            'calendar': cal,
            'weekdays': weekdays,
            'weekends': weekends,
            'total_days': weekdays + weekends,
            'first_day': first_day.strftime("%A"),
            'last_day': last_day.strftime("%A")
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'calendar': None
        }

# Register calendar functions
def register_calendar_functions():
    """Register calendar functions with the function registry."""
    registry = get_function_registry()
    
    # Current time function
    registry.register_function(
        definition=FunctionDefinition(
            name="get_current_time",
            description="Get current date and time in various formats",
            function_type=FunctionType.CALENDAR,
            parameters=[
                FunctionParameter(
                    name="format",
                    type=ParameterType.STRING,
                    description="Time format string or special format ('iso', 'timestamp')",
                    required=False,
                    default="%Y-%m-%d %H:%M:%S",
                    enum_values=["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%H:%M:%S", "iso", "timestamp"]
                ),
                FunctionParameter(
                    name="timezone_name",
                    type=ParameterType.STRING,
                    description="Timezone name",
                    required=False,
                    default="UTC"
                )
            ],
            examples=[
                {"format": "%Y-%m-%d", "result": "2024-01-15"},
                {"format": "iso", "result": "2024-01-15T14:30:00Z"}
            ]
        ),
        handler=get_current_time
    )
    
    # Add time function
    registry.register_function(
        definition=FunctionDefinition(
            name="add_time",
            description="Add time duration to a base datetime",
            function_type=FunctionType.CALENDAR,
            parameters=[
                FunctionParameter(
                    name="base_time",
                    type=ParameterType.STRING,
                    description="Base time in ISO format or timestamp",
                    required=True
                ),
                FunctionParameter(
                    name="days",
                    type=ParameterType.INTEGER,
                    description="Days to add",
                    required=False,
                    default=0
                ),
                FunctionParameter(
                    name="hours",
                    type=ParameterType.INTEGER,
                    description="Hours to add",
                    required=False,
                    default=0
                ),
                FunctionParameter(
                    name="minutes",
                    type=ParameterType.INTEGER,
                    description="Minutes to add",
                    required=False,
                    default=0
                ),
                FunctionParameter(
                    name="seconds",
                    type=ParameterType.INTEGER,
                    description="Seconds to add",
                    required=False,
                    default=0
                )
            ],
            agent_restrictions=["meetings", "general", "finance"]
        ),
        handler=add_time
    )
    
    # Business days calculation
    registry.register_function(
        definition=FunctionDefinition(
            name="get_business_days",
            description="Calculate business days between two dates",
            function_type=FunctionType.CALENDAR,
            parameters=[
                FunctionParameter(
                    name="start_date",
                    type=ParameterType.STRING,
                    description="Start date in YYYY-MM-DD format",
                    required=True,
                    pattern=r"^\d{4}-\d{2}-\d{2}$"
                ),
                FunctionParameter(
                    name="end_date",
                    type=ParameterType.STRING,
                    description="End date in YYYY-MM-DD format",
                    required=True,
                    pattern=r"^\d{4}-\d{2}-\d{2}$"
                )
            ],
            agent_restrictions=["finance", "meetings", "general"]
        ),
        handler=get_business_days
    )
    
    # Calendar info function
    registry.register_function(
        definition=FunctionDefinition(
            name="get_calendar_info",
            description="Get calendar information for a specific month",
            function_type=FunctionType.CALENDAR,
            parameters=[
                FunctionParameter(
                    name="year",
                    type=ParameterType.INTEGER,
                    description="Year (e.g., 2024)",
                    required=True,
                    min_value=1900,
                    max_value=2100
                ),
                FunctionParameter(
                    name="month",
                    type=ParameterType.INTEGER,
                    description="Month (1-12)",
                    required=True,
                    min_value=1,
                    max_value=12
                )
            ],
            agent_restrictions=["meetings", "general"]
        ),
        handler=get_calendar_info
    )
