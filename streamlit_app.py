# streamlit_app.py — ARIASKA Advanced Visualization Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import sleep

# Configure page
st.set_page_config(
    page_title="ARIASKA_RL Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🧠 ARIASKA Multi-Agent RL Platform")
st.markdown("Real-time monitoring and visualization of the ARIASKA_RL cybersecurity training system")

# Add a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Training Overview", "Agent Performance", "GPT Usage", "Environment", "Memory Analysis"]
)

# Constants
LOG_DIR = "logs"
MODELS_DIR = "models"
MEMORY_DIR = "core/memories"

# Function to load data (implement as needed based on actual data format)
def load_training_data():
    """Load training data from logs directory"""
    try:
        # Search for training metrics file
        metrics_path = os.path.join(MODELS_DIR, "training_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                return json.load(f)
        else:
            # Try to find any JSON logs
            json_files = list(Path(LOG_DIR).glob("**/*.json"))
            if json_files:
                with open(json_files[0], "r") as f:
                    return json.load(f)
            else:
                return {
                    "episode_rewards": [],
                    "episode_steps": [],
                    "episode_phases": [],
                    "unique_actions": [],
                    "timestamps": []
                }
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return {
            "episode_rewards": [],
            "episode_steps": [],
            "episode_phases": [],
            "unique_actions": [],
            "timestamps": []
        }

def load_gpt_usage():
    """Load GPT usage statistics"""
    try:
        # Look for GPT usage logs
        usage_path = os.path.join(LOG_DIR, "gpt_usage.json")
        if os.path.exists(usage_path):
            with open(usage_path, "r") as f:
                return json.load(f)
        else:
            # Generate sample data if not found
            return {
                "by_model": {"gpt-4o-mini": 1000, "gpt-4.1-nano": 500},
                "by_task": {"reasoning": 800, "decision": 400, "embedding": 300},
                "by_agent": {"RedAgent": 1200, "BlueAgent": 800, "OrionAgent": 500},
                "total": 2500,
                "fallbacks": {"gpt-4o-mini": 5}
            }
    except Exception as e:
        st.error(f"Error loading GPT usage data: {e}")
        return {"by_model": {}, "by_task": {}, "by_agent": {}, "total": 0, "fallbacks": {}}

def load_agent_actions():
    """Load agent actions history"""
    try:
        actions_path = os.path.join(LOG_DIR, "agent_actions.json")
        if os.path.exists(actions_path):
            with open(actions_path, "r") as f:
                return json.load(f)
        else:
            # Generate sample data
            return {
                "RedAgent": ["nmap -sV", "gobuster dir", "hydra -l admin"],
                "BlueAgent": ["monitor traffic", "reset firewall", "deploy honeypot"],
                "OrionAgent": ["strategic assessment", "adjust curriculum", "evaluate performance"]
            }
    except Exception as e:
        st.error(f"Error loading agent actions: {e}")
        return {}

# Load data
training_data = load_training_data()
gpt_usage = load_gpt_usage()
agent_actions = load_agent_actions()

# Training Overview page
if page == "Training Overview":
    st.header("📊 Training Overview")
    
    # Display key metrics in a dashboard
    col1, col2, col3 = st.columns(3)
    
    episode_rewards = training_data.get("episode_rewards", [])
    episode_steps = training_data.get("episode_steps", [])
    
    with col1:
        st.metric("Total Episodes", len(episode_rewards))
        
    with col2:
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        st.metric("Average Reward", f"{avg_reward:.2f}")
        
    with col3:
        avg_steps = np.mean(episode_steps) if episode_steps else 0
        st.metric("Average Steps per Episode", f"{avg_steps:.1f}")
    
    # Plot training progress
    st.subheader("Training Progress")
    
    if episode_rewards:
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            "Episode": range(1, len(episode_rewards) + 1),
            "Reward": episode_rewards,
            "Steps": episode_steps
        })
        
        # Plot reward trend
        fig = px.line(
            df, 
            x="Episode", 
            y="Reward", 
            title="Reward per Episode",
            markers=True
        )
        fig.update_layout(
            xaxis_title="Episode",
            yaxis_title="Reward",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot steps per episode
        fig2 = px.line(
            df, 
            x="Episode", 
            y="Steps", 
            title="Steps per Episode",
            markers=True
        )
        fig2.update_layout(
            xaxis_title="Episode",
            yaxis_title="Steps",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No training data available yet. Start training to see metrics.")

# Agent Performance page
elif page == "Agent Performance":
    st.header("🤖 Agent Performance")
    
    # Select agent to view
    agent = st.selectbox("Select Agent", ["RedAgent", "BlueAgent", "ScoutAgent", "ShadowAgent", "OrionAgent"])
    
    # Display agent-specific metrics
    st.subheader(f"{agent} Performance Metrics")
    
    # Sample agent metrics (replace with actual data loading)
    agent_metrics = {
        "success_rate": 0.65,
        "detection_rate": 0.20,
        "avg_reward": 12.5,
        "phase_distribution": {"recon": 30, "enumeration": 25, "exploit": 20, "privesc": 15, "exfiltrate": 10}
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Success Rate", f"{agent_metrics['success_rate']:.0%}")
        
    with col2:
        st.metric("Detection Rate", f"{agent_metrics['detection_rate']:.0%}")
        
    with col3:
        st.metric("Avg. Reward", f"{agent_metrics['avg_reward']:.1f}")
    
    # Phase distribution
    st.subheader("Phase Distribution")
    
    phase_df = pd.DataFrame(
        {"Phase": list(agent_metrics["phase_distribution"].keys()),
         "Count": list(agent_metrics["phase_distribution"].values())}
    )
    
    fig = px.pie(
        phase_df,
        values="Count",
        names="Phase",
        title="Time Spent in Each Phase",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent actions
    st.subheader("Recent Actions")
    
    if agent in agent_actions and agent_actions[agent]:
        for i, action in enumerate(reversed(agent_actions[agent][:10])):
            st.code(action, language="bash")
    else:
        st.info(f"No actions recorded for {agent} yet.")
    
    # Display exploration rate over time
    st.subheader("Exploration Rate (Epsilon)")
    
    # Sample epsilon data (replace with actual data loading)
    epsilon_data = [1.0, 0.95, 0.9, 0.85, 0.8, 0.76, 0.72, 0.68, 0.65, 0.62]
    fig = px.line(
        x=list(range(len(epsilon_data))),
        y=epsilon_data,
        labels={'x': 'Episode', 'y': 'Epsilon'},
        title='Exploration Rate Over Time'
    )
    st.plotly_chart(fig, use_container_width=True)

# GPT Usage page
elif page == "GPT Usage":
    st.header("🧠 GPT Usage Analytics")
    
    # Display GPT usage metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tokens", gpt_usage["total"])
        
    with col2:
        fallbacks = sum(gpt_usage["fallbacks"].values()) if gpt_usage["fallbacks"] else 0
        st.metric("Total Fallbacks", fallbacks)
        
    with col3:
        models = list(gpt_usage["by_model"].keys())
        primary_model = models[0] if models else "N/A"
        st.metric("Primary Model", primary_model)
    
    # Usage by model pie chart
    st.subheader("Token Usage by Model")
    
    model_df = pd.DataFrame({
        "Model": list(gpt_usage["by_model"].keys()),
        "Tokens": list(gpt_usage["by_model"].values())
    })
    
    fig = px.pie(
        model_df,
        values="Tokens",
        names="Model",
        title="Token Distribution by Model",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Usage by agent bar chart
    st.subheader("Token Usage by Agent")
    
    agent_df = pd.DataFrame({
        "Agent": list(gpt_usage["by_agent"].keys()),
        "Tokens": list(gpt_usage["by_agent"].values())
    })
    
    fig = px.bar(
        agent_df,
        x="Agent",
        y="Tokens",
        title="Token Usage by Agent",
        color="Agent",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Usage by task
    st.subheader("Token Usage by Task Type")
    
    task_df = pd.DataFrame({
        "Task": list(gpt_usage["by_task"].keys()),
        "Tokens": list(gpt_usage["by_task"].values())
    })
    
    fig = px.bar(
        task_df,
        x="Task",
        y="Tokens",
        title="Token Usage by Task Type",
        color="Task"
    )
    st.plotly_chart(fig, use_container_width=True)

# Environment page
elif page == "Environment":
    st.header("🌐 Environment Status")
    
    # Environment type
    env_type = "Simulated"  # Replace with actual environment type
    
    st.info(f"Current Environment Mode: {env_type}")
    
    # Environment state visualization
    st.subheader("Current Environment State")
    
    # Sample environment state (replace with actual state loading)
    env_state = {
        "phase": "exploit",
        "open_ports": [22, 80, 443, 3306],
        "services": ["ssh", "http", "https", "mysql"],
        "blue_team_alert": 45.5,
        "detection_risk": 6.2,
        "privilege_level": "user",
        "credentials_found": True,
        "data_exfiltrated": False
    }
    
    # Display state as a table
    st.json(env_state)
    
    # Display alert level gauge
    st.subheader("Blue Team Alert Level")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = env_state["blue_team_alert"],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Alert Level"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Network visualization
    st.subheader("Network Visualization")
    
    # Sample network data (replace with actual data)
    network_data = {
        "nodes": [
            {"id": "target", "label": "Target Host", "group": 1},
            {"id": "port22", "label": "Port 22", "group": 2},
            {"id": "port80", "label": "Port 80", "group": 2},
            {"id": "port443", "label": "Port 443", "group": 2},
            {"id": "ssh", "label": "SSH Service", "group": 3},
            {"id": "http", "label": "HTTP Service", "group": 3},
            {"id": "https", "label": "HTTPS Service", "group": 3}
        ],
        "links": [
            {"source": "target", "target": "port22"},
            {"source": "target", "target": "port80"},
            {"source": "target", "target": "port443"},
            {"source": "port22", "target": "ssh"},
            {"source": "port80", "target": "http"},
            {"source": "port443", "target": "https"}
        ]
    }
    
    # Display placeholder for network visualization
    st.info("Network visualization will be available in future updates. Currently showing environment state in table format.")

# Memory Analysis page
elif page == "Memory Analysis":
    st.header("🧠 Memory System Analysis")
    
    # Memory system metrics
    memory_metrics = {
        "total_memories": 1250,
        "red_agent_memories": 650,
        "blue_agent_memories": 400,
        "orion_memories": 200,
        "deduplicated": 120,
        "high_priority_memories": 85
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Memories", memory_metrics["total_memories"])
        
    with col2:
        st.metric("Deduplicated", memory_metrics["deduplicated"])
        
    with col3:
        st.metric("High Priority", memory_metrics["high_priority_memories"])
    
    # Memory distribution by agent
    st.subheader("Memory Distribution by Agent")
    
    memory_df = pd.DataFrame({
        "Agent": ["RedAgent", "BlueAgent", "OrionAgent"],
        "Memories": [memory_metrics["red_agent_memories"], 
                    memory_metrics["blue_agent_memories"], 
                    memory_metrics["orion_memories"]]
    })
    
    fig = px.pie(
        memory_df,
        values="Memories",
        names="Agent",
        title="Memory Distribution",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Memory utilization over time
    st.subheader("Memory Utilization Over Time")
    
    # Sample memory growth data (replace with actual data)
    memory_growth = [100, 220, 350, 500, 680, 800, 950, 1050, 1150, 1250]
    
    fig = px.line(
        x=list(range(1, len(memory_growth) + 1)),
        y=memory_growth,
        labels={'x': 'Episode', 'y': 'Total Memories'},
        title='Memory Growth Over Episodes'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Memory sample explorer
    st.subheader("Memory Sample Explorer")
    
    # Agent selector for memory samples
    memory_agent = st.selectbox("Select Agent for Memory Samples", 
                               ["RedAgent", "BlueAgent", "ScoutAgent", "OrionAgent"])
    
    # Sample memories (replace with actual memory loading)
    sample_memories = [
        {"state": "recon", "action": "nmap -sV 10.10.10.10", "reward": 5.5, "priority": "high"},
        {"state": "enumeration", "action": "gobuster dir -u http://10.10.10.10", "reward": 3.2, "priority": "medium"},
        {"state": "exploit", "action": "sqlmap -u 'http://10.10.10.10/page.php?id=1'", "reward": 12.8, "priority": "high"}
    ]
    
    for i, memory in enumerate(sample_memories):
        with st.expander(f"Memory {i+1}: {memory['action']}"):
            st.json(memory)

# Add a refresh button that doesn't reload the page but updates data
if st.button("Refresh Data"):
    training_data = load_training_data()
    gpt_usage = load_gpt_usage()
    agent_actions = load_agent_actions()
    st.success("Data refreshed!")
    
# Add footer with timestamp
st.sidebar.markdown("---")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.markdown("ARIASKA_RL Platform v2.1 APEX")
