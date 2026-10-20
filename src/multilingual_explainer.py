"""
English-Only Dual-Audience (Technical vs. Everyday User) Explanation Engine.

Translates machine learning attributions and sensor values into:
- 'simple': Everyday conversational English for EV drivers (clear metaphors, no jargon)
- 'technical': Battery engineering terminology with units, gradients, and formal BMS actions
"""

from typing import Any, Dict, List

FEATURE_EXPLANATIONS = {
    "temp_max": {
        "simple": "Battery temperature is higher than normal",
        "technical": "Peak discharge cell temperature (temp_max)",
    },
    "temp_mean": {
        "simple": "The battery is running warm consistently",
        "technical": "Average discharge cycle thermal profile (temp_mean)",
    },
    "temp_trend": {
        "simple": "The battery is heating up faster with each cycle",
        "technical": "Positive thermal gradient across recent cycles (temp_trend)",
    },
    "temp_change_rate": {
        "simple": "Sudden sharp rise in battery heat",
        "technical": "Elevated rate of temperature change (temp_change_rate)",
    },
    "capacity_fade_pct": {
        "simple": "Battery range and usable storage have degraded noticeably",
        "technical": "Cumulative capacity loss relative to rated nominal (capacity_fade_pct)",
    },
    "capacity_fade_rate": {
        "simple": "The battery is losing its charge-holding ability unusually fast",
        "technical": "Accelerated degradation slope in discharge capacity (capacity_fade_rate)",
    },
    "impedance_rise_pct": {
        "simple": "Internal resistance is high, making the battery struggle to deliver power smoothly",
        "technical": "Cumulative internal cell impedance rise (impedance_rise_pct)",
    },
    "voltage_deviation": {
        "simple": "Unstable electrical voltage detected across the cell",
        "technical": "Voltage dispersion deviating from expected operating baseline (voltage_deviation)",
    },
    "voltage_drop": {
        "simple": "Voltage drops faster than expected while driving under load",
        "technical": "Steep voltage collapse gradient during discharge cycle (voltage_drop)",
    },
    "re": {
        "simple": "Electrolyte aging inside the battery cells",
        "technical": "Ohmic electrolyte bulk resistance (Re)",
    },
    "rct": {
        "simple": "Sluggish electrochemical reaction, causing power lag",
        "technical": "Interfacial charge transfer resistance (Rct)",
    },
    "duration": {
        "simple": "Battery drained significantly earlier than rated",
        "technical": "Shortened discharge duty cycle duration (duration)",
    },
    "soc_estimated": {
        "simple": "Estimated remaining usable energy is critically low",
        "technical": "Computed state of charge metric (soc_estimated)",
    },
}

ACTIONS = {
    "LOW": {
        "simple": "Your battery is operating safely. Continue normal driving and charging.",
        "technical": "Nominal operating parameters confirmed. Maintain standard BMS telemetry logging.",
    },
    "MEDIUM": {
        "simple": "Early wear noticed. Avoid fast charging in direct heat and monitor battery range.",
        "technical": "Moderate degradation detected. Increase telemetry polling frequency and avoid elevated C-rate charging.",
    },
    "HIGH": {
        "simple": "Warning: High battery stress! Slow down charging, let the pack cool, and visit an EV service center soon.",
        "technical": "Elevated battery risk alert. Throttle charge current, enforce active thermal management, and schedule diagnostic inspection.",
    },
    "CRITICAL": {
        "simple": "DANGER: Critical battery fault! Stop vehicle or disconnect charger immediately. Contact emergency service.",
        "technical": "CRITICAL RISK EXCEEDED. Trip automatic circuit breaker, cease charging immediately, and trigger emergency BMS isolation protocol.",
    },
}

RISK_LABELS = {
    "LOW": "Safe / Normal",
    "MEDIUM": "Moderate Attention",
    "HIGH": "High Warning",
    "CRITICAL": "Critical Danger",
}


def translate_warning(
    warning: Dict[str, Any],
    mode: str = "simple",
) -> Dict[str, Any]:
    """
    Generate an audience-targeted English warning explanation.

    Parameters
    ----------
    warning : dict
        Standard warning dict output from generate_warning().
    mode : str
        'simple' (everyday driver/user) or 'technical' (battery engineer).

    Returns
    -------
    dict
        Audience-tailored representation of the warning.
    """
    risk_level = warning.get("risk_level", "LOW").upper()
    factors = warning.get("contributing_factors", [])

    factor_phrases = []
    for f in factors[:3]:
        feat_name = f.get("feature", "")
        lookup = FEATURE_EXPLANATIONS.get(feat_name, {})
        phrase = lookup.get(mode)
        if not phrase:
            phrase = f.get("description", feat_name)

        if mode == "technical":
            val = f.get("actual_value", 0.0)
            unit = f.get("unit", "")
            shap_c = f.get("shap_contribution", f.get("shap_value", 0.0))
            phrase = f"{phrase} [Value: {val:.2f} {unit}, Impact: {shap_c:+.3f}]"

        factor_phrases.append(phrase)

    action_text = ACTIONS.get(risk_level, ACTIONS["LOW"]).get(mode, "")
    risk_name = RISK_LABELS.get(risk_level, risk_level)

    if mode == "simple":
        driver_str = "; ".join(factor_phrases) if factor_phrases else "General operating state"
        summary_msg = f"Battery Status: {risk_name}. What is happening: {driver_str}. Next Steps: {action_text}"
    else:
        driver_str = "; ".join(factor_phrases) if factor_phrases else "Monitored telemetry"
        summary_msg = f"Risk Classification: {risk_name} (Score: {warning.get('risk_score', 0):.3f}). Primary SHAP Contributors: {driver_str}. Recommended BMS Mitigation: {action_text}"

    return {
        "mode": mode,
        "risk_level_localized": risk_name,
        "summary_message": summary_msg,
        "action": action_text,
        "explained_factors": factor_phrases,
    }
