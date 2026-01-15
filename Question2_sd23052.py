import streamlit as st
import operator
from typing import List, Dict, Any, Tuple

# ============================ #
# RULE ENGINE CONFIGURATION #
# ============================ #
COMPARISONS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

DEFAULT_CONDITIONS: List[Dict[str, Any]] = [
    {
        "name": "Windows open → turn AC off",
        "priority": 100,
        "conditions": [
            ["windows_open", "==", True]
        ],
        "action": {
            "ac_mode": "OFF",
            "fan_speed": "LOW",
            "setpoint": None,
            "reason": "Windows are open"
        }
    },
    {
        "name": "No one home → eco mode",
        "priority": 90,
        "conditions": [
            ["occupancy", "==", "EMPTY"],
            ["temperature", ">=", 24]
        ],
        "action": {
            "ac_mode": "ECO",
            "fan_speed": "LOW",
            "setpoint": 27,
            "reason": "Home empty; save energy"
        }
    },
    {
        "name": "Hot & humid (occupied) → cool strong",
        "priority": 80,
        "conditions": [
            ["occupancy", "==", "OCCUPIED"],
            ["temperature", ">=", 30],
            ["humidity", ">=", 70]
        ],
        "action": {
            "ac_mode": "COOL",
            "fan_speed": "HIGH",
            "setpoint": 23,
            "reason": "Hot and humid"
        }
    },
    {
        "name": "Hot (occupied) → cool",
        "priority": 70,
        "conditions": [
            ["occupancy", "==", "OCCUPIED"],
            ["temperature", ">=", 28]
        ],
        "action": {
            "ac_mode": "COOL",
            "fan_speed": "MEDIUM",
            "setpoint": 24,
            "reason": "Temperature high"
        }
    },
    {
        "name": "Slightly warm (occupied) → gentle cool",
        "priority": 60,
        "conditions": [
            ["occupancy", "==", "OCCUPIED"],
            ["temperature", ">=", 26],
            ["temperature", "<", 28]
        ],
        "action": {
            "ac_mode": "COOL",
            "fan_speed": "LOW",
            "setpoint": 25,
            "reason": "Slightly warm"
        }
    },
    {
        "name": "Night (occupied) → sleep mode",
        "priority": 75,
        "conditions": [
            ["occupancy", "==", "OCCUPIED"],
            ["time_of_day", "==", "NIGHT"],
            ["temperature", ">=", 26]
        ],
        "action": {
            "ac_mode": "SLEEP",
            "fan_speed": "LOW",
            "setpoint": 26,
            "reason": "Night comfort"
        }
    },
    {
        "name": "Too cold → turn off",
        "priority": 85,
        "conditions": [
            ["temperature", "<=", 22]
        ],
        "action": {
            "ac_mode": "OFF",
            "fan_speed": "LOW",
            "setpoint": None,
            "reason": "Already cold"
        }
    }
]

def evaluate_condition(facts: Dict[str, Any], condition: List[Any]) -> bool:
    field, operator, value = condition
    if field not in facts or operator not in COMPARISONS:
        return False
    return COMPARISONS[operator](facts[field], value)

def check_rule_match(facts: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    return all(evaluate_condition(facts, cond) for cond in rule["conditions"])

def execute_rules(facts: Dict[str, Any], rules: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    triggered_rules = [rule for rule in rules if check_rule_match(facts, rule)]
    if not triggered_rules:
        return {"ac_mode": "OFF", "reason": "No matching rules"}, []
    
    sorted_rules = sorted(triggered_rules, key=lambda r: r["priority"], reverse=True)
    return sorted_rules[0]["action"], sorted_rules

# ============================== #
# STREAMLIT INTERFACE #
# ============================== #
st.set_page_config(page_title="Smart AC System", layout="wide")
st.title("Smart Air Conditioner Rule-Based System")

# Sidebar for input
with st.sidebar:
    st.header("Current Home Conditions")
    temperature_input = st.number_input("Temperature (°C)", value=22)
    humidity_input = st.number_input("Humidity (%)", value=46)
    occupancy_input = st.selectbox("Occupancy", ["OCCUPIED", "EMPTY"])
    time_of_day_input = st.selectbox("Time of Day", ["DAY", "NIGHT"])
    windows_open_input = st.checkbox("Windows Open", value=False)
    evaluate_button = st.button("Evaluate AC Action")

# Facts data based on user input
facts_data = {
    "temperature": temperature_input,
    "humidity": humidity_input,
    "occupancy": occupancy_input,
    "time_of_day": time_of_day_input,
    "windows_open": windows_open_input
}

# Display the facts entered
st.subheader("Input Facts")
st.json(facts_data)

# If the button is clicked, run the rule-based system
if evaluate_button:
    action_details, matching_rules = execute_rules(facts_data, DEFAULT_CONDITIONS)

    # Display the recommended AC action
    st.subheader("Recommended AC Action")
    st.success(
        f"""
        **AC Mode:** {action_details['ac_mode']}  
        **Fan Speed:** {action_details['fan_speed']}  
        **Setpoint:** {action_details['setpoint']}  
        **Reason:** {action_details['reason']}
        """
    )

    # Display matched rules
    st.subheader("Matched Rules (Ordered by Priority)")
    for rule in matching_rules:
        st.write(f"• **{rule['name']}** (Priority: {rule['priority']})")
