"""
Visualization of Three-Tier Risk Management Reward Function
============================================================

This script generates visualizations of the reward function
to help understand the penalty/reward structure.

Usage:
    python docs/20251024/reward_function_visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def calculate_exit_reward(pnl_usdt, profit_aim=25):
    """
    Calculate exit reward based on three-tier penalty system.
    
    Args:
        pnl_usdt: PNL in USDT (can be positive or negative)
        profit_aim: Profit aim in USDT (default 25 USDT = 2.5% leveraged)
    
    Returns:
        float: Reward value
    """
    if pnl_usdt > 0:
        # Profit tiers (simplified - actual implementation includes multipliers)
        base_factor = 15.0
        if pnl_usdt > profit_aim * 10:
            return pnl_usdt / 100 * base_factor * 32  # mega_win
        elif pnl_usdt > profit_aim * 5:
            return pnl_usdt / 100 * base_factor * 16  # huge_win
        elif pnl_usdt > profit_aim * 3:
            return pnl_usdt / 100 * base_factor * 8   # big_win
        elif pnl_usdt > profit_aim * 1.5:
            return pnl_usdt / 100 * base_factor * 4   # medium_win
        elif pnl_usdt > profit_aim:
            return pnl_usdt / 100 * base_factor * 2   # small_win
        else:
            return pnl_usdt / 100 * base_factor       # base profit
    else:
        # Loss tiers
        abs_pnl = abs(pnl_usdt)
        
        if abs_pnl < profit_aim:  # Safe zone: 0 to -25 USDT
            return -5
        elif abs_pnl < profit_aim * 2:  # Warning zone: -25 to -50 USDT
            ratio = (abs_pnl - profit_aim) / profit_aim
            return -5 + (-50 - (-5)) * ratio
        elif abs_pnl < profit_aim * 3:  # Danger zone: -50 to -75 USDT
            ratio = (abs_pnl - profit_aim * 2) / profit_aim
            return -50 + (-150 - (-50)) * ratio
        else:  # Extreme loss or liquidation
            return -200


def calculate_holding_feedback(pnl_usdt, profit_aim=25):
    """
    Calculate immediate feedback while holding position.
    
    Args:
        pnl_usdt: Unrealized PNL in USDT
        profit_aim: Profit aim in USDT
    
    Returns:
        float: Feedback value per timestep
    """
    if pnl_usdt > 0:
        return pnl_usdt / 100 * 0.3
    else:
        abs_pnl = abs(pnl_usdt)
        
        if abs_pnl < profit_aim:  # Safe zone
            return pnl_usdt / 100 * 0.3
        elif abs_pnl < profit_aim * 2:  # Warning zone
            return pnl_usdt / 100 * 1.5
        else:  # Danger zone
            return pnl_usdt / 100 * 3.0


def plot_exit_reward_function():
    """Generate exit reward function plot."""
    pnl_range = np.linspace(-100, 300, 1000)
    rewards = [calculate_exit_reward(pnl) for pnl in pnl_range]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot main reward function
    ax.plot(pnl_range, rewards, linewidth=2.5, color='#2E86AB', label='Exit Reward Function')
    
    # Add zone backgrounds
    ax.axvspan(-100, 0, alpha=0.1, color='red', label='Loss Territory')
    ax.axvspan(0, 300, alpha=0.1, color='green', label='Profit Territory')
    
    # Add risk zone markers
    ax.axvline(-25, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Stop Loss (-25 USDT)')
    ax.axvline(-50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Warning Boundary (-50 USDT)')
    ax.axvline(-75, color='darkred', linestyle='--', linewidth=2, alpha=0.8, label='Liquidation Threshold (-75 USDT)')
    
    # Add horizontal reference lines
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axhline(-5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(-50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(-150, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(-200, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Annotations for key points
    ax.annotate('Safe Zone\n(0 to -25 USDT)\nPenalty: -5',
                xy=(-12.5, -5), xytext=(-12.5, -80),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green'))
    
    ax.annotate('Warning Zone\n(-25 to -50 USDT)\nPenalty: -5 to -50',
                xy=(-37.5, -27.5), xytext=(-37.5, -120),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='orange'))
    
    ax.annotate('Danger Zone\n(-50 to -75 USDT)\nPenalty: -50 to -150',
                xy=(-62.5, -100), xytext=(-62.5, -160),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))
    
    ax.annotate('Liquidation\n(< -75 USDT)\nPenalty: -200',
                xy=(-85, -200), xytext=(-90, -230),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkred'))
    
    # Profit annotations
    ax.annotate('Small Win\n(+25 to +37.5 USDT)\n2x multiplier',
                xy=(31, 15), xytext=(31, 60),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='blue'))
    
    ax.annotate('Mega Win\n(> +250 USDT)\n32x multiplier',
                xy=(275, 120), xytext=(200, 180),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='purple'))
    
    # Styling
    ax.set_xlabel('PNL (USDT)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
    ax.set_title('Three-Tier Risk Management: Exit Reward Function\n10x Leverage | 100 USDT Position | Profit Aim: 25 USDT',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)
    
    # Set axis limits
    ax.set_xlim(-100, 300)
    ax.set_ylim(-250, 200)
    
    plt.tight_layout()
    plt.savefig('docs/20251024/exit_reward_function.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: docs/20251024/exit_reward_function.png")
    plt.show()


def plot_holding_feedback():
    """Generate holding feedback plot."""
    pnl_range = np.linspace(-100, 100, 1000)
    feedback = [calculate_holding_feedback(pnl) for pnl in pnl_range]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot feedback function
    ax.plot(pnl_range, feedback, linewidth=2.5, color='#A23B72', label='Holding Feedback')
    
    # Add zone backgrounds
    colors = ['lightcoral', 'lightyellow', 'lightgreen']
    zones = [(-100, -50), (-50, -25), (-25, 0), (0, 100)]
    zone_names = ['Danger\n3.0x feedback', 'Warning\n1.5x feedback', 
                  'Safe\n0.3x feedback', 'Profit\n0.3x feedback']
    
    for i, ((start, end), name) in enumerate(zip(zones, zone_names)):
        if i == 0:
            color = 'red'
            alpha = 0.2
        elif i == 1:
            color = 'orange'
            alpha = 0.15
        else:
            color = 'green'
            alpha = 0.1
        
        ax.axvspan(start, end, alpha=alpha, color=color)
        mid_point = (start + end) / 2
        ax.text(mid_point, ax.get_ylim()[1] * 0.9, name,
               ha='center', va='top', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add reference lines
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(-25, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(-50, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(0, color='green', linestyle='-', linewidth=1, alpha=0.5)
    
    # Annotations
    ax.annotate('Escalating negative feedback\npushes agent to exit',
                xy=(-60, -1.8), xytext=(-75, -2.5),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5', color='red', lw=2))
    
    ax.annotate('Gentle reward encourages\nholding winning positions',
                xy=(50, 0.15), xytext=(65, 0.8),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.5', color='green', lw=2))
    
    # Styling
    ax.set_xlabel('Unrealized PNL (USDT)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Immediate Feedback (per timestep)', fontsize=14, fontweight='bold')
    ax.set_title('Dynamic Holding Feedback: Risk-Aware Position Management\nEncourages holding profits, discourages holding losses',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('docs/20251024/holding_feedback_function.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: docs/20251024/holding_feedback_function.png")
    plt.show()


def plot_comparison_table():
    """Generate comparison table of old vs new system."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Table data
    scenarios = [
        ['Scenario', 'PNL (USDT)', 'Old Reward', 'New Reward', 'Change', 'Rationale'],
        ['Small Profit', '+25', '+360', '+0.75-1.5', '↓ -99.6%', 'Prevent reward explosion'],
        ['Medium Profit', '+50', '+600', '+3.0-6.0', '↓ -99.0%', 'Maintain relative value'],
        ['Large Profit', '+100', '+960', '+12-24', '↓ -97.5%', 'Scale down proportionally'],
        ['Mega Profit', '+250', '+3600', '+60-120', '↓ -96.7%', 'Keep tier structure'],
        ['', '', '', '', '', ''],
        ['Small Loss', '-10', '-100', '-5', '↑ +95%', 'Accept small losses'],
        ['Stop Loss', '-25', '-250', '-5', '↑ +98%', 'No penalty at stop loss'],
        ['Warning', '-37.5', '-375', '-27.5', '↑ +92.7%', 'Medium penalty'],
        ['Danger Low', '-50', '-500', '-50', '↑ +90%', 'Heavy warning'],
        ['Danger High', '-62.5', '-625', '-100', '↑ +84%', 'Approaching liquidation'],
        ['Near Liquidation', '-75', '-750', '-150', '↑ +80%', 'Last chance to exit'],
        ['Liquidation', '-80', '-1000', '-200', '↑ +80%', 'Forced closure'],
    ]
    
    # Color coding for changes
    colors = [
        ['#E8E8E8'] * 6,  # Header
        ['#C8E6C9'] * 6,  # Small profit
        ['#A5D6A7'] * 6,  # Medium profit
        ['#81C784'] * 6,  # Large profit
        ['#66BB6A'] * 6,  # Mega profit
        ['#FFFFFF'] * 6,  # Separator
        ['#FFF9C4'] * 6,  # Small loss
        ['#FFF59D'] * 6,  # Stop loss
        ['#FFEB3B'] * 6,  # Warning
        ['#FFD54F'] * 6,  # Danger low
        ['#FFCA28'] * 6,  # Danger high
        ['#FFB300'] * 6,  # Near liquidation
        ['#FF6F00'] * 6,  # Liquidation
    ]
    
    # Create table
    table = ax.table(cellText=scenarios, cellColours=colors,
                    loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(6):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', size=12)
        cell.set_facecolor('#607D8B')
        cell.set_text_props(color='white')
    
    # Bold scenario names
    for i in range(1, len(scenarios)):
        cell = table[(i, 0)]
        cell.set_text_props(weight='bold')
    
    # Title
    ax.text(0.5, 0.98, 'Reward System Comparison: Old vs New',
           ha='center', va='top', fontsize=18, fontweight='bold',
           transform=ax.transAxes)
    
    ax.text(0.5, 0.94, 'Three-Tier Risk Management with Normalized Rewards',
           ha='center', va='top', fontsize=14, style='italic',
           transform=ax.transAxes, color='#555555')
    
    # Summary text
    summary = """
    Key Improvements:
    
    ✅ Reward scale reduced by 95-99% to prevent value function explosion
    ✅ Relative reward structure maintained (profit tiers still incentivized)
    ✅ Loss penalties reduced by 80-98% in safe/warning zones
    ✅ Three-tier system teaches proper risk management
    ✅ Liquidation penalty remains strong but proportional (-200 vs old -1000)
    """
    
    ax.text(0.5, 0.05, summary,
           ha='center', va='bottom', fontsize=11,
           transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=1', facecolor='#F5F5F5', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('docs/20251024/reward_comparison_table.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: docs/20251024/reward_comparison_table.png")
    plt.show()


def plot_risk_zones_diagram():
    """Generate risk zones visual diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define zones
    zones = [
        {'name': 'Profit Zone', 'range': (0, 300), 'color': '#4CAF50', 'y': 0.8},
        {'name': '🟢 Safe Zone', 'range': (-25, 0), 'color': '#8BC34A', 'y': 0.6},
        {'name': '🟡 Warning Zone', 'range': (-50, -25), 'color': '#FFC107', 'y': 0.4},
        {'name': '🔴 Danger Zone', 'range': (-75, -50), 'color': '#FF9800', 'y': 0.2},
        {'name': '⚫ Liquidation', 'range': (-100, -75), 'color': '#F44336', 'y': 0.0},
    ]
    
    # Draw zones
    for zone in zones:
        start, end = zone['range']
        width = end - start
        height = 0.15
        y_pos = zone['y']
        
        rect = Rectangle((start, y_pos), width, height,
                        facecolor=zone['color'], edgecolor='black',
                        linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Add zone name
        mid_x = (start + end) / 2
        ax.text(mid_x, y_pos + height/2, zone['name'],
               ha='center', va='center', fontsize=14, fontweight='bold',
               color='white', bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='black', alpha=0.5))
        
        # Add range text
        if 'Profit' not in zone['name']:
            ax.text(mid_x, y_pos - 0.05, f"{start} to {end} USDT",
                   ha='center', va='top', fontsize=10, style='italic')
    
    # Add key thresholds
    thresholds = [
        (-25, 'Stop Loss\n-25 USDT\n(-2.5% price move)'),
        (-50, 'Warning Boundary\n-50 USDT\n(-5.0% price move)'),
        (-75, 'Liquidation Threshold\n-75 USDT\n(-7.5% price move)'),
    ]
    
    for x, label in thresholds:
        ax.axvline(x, color='black', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(x, 1.05, label, ha='center', va='bottom', fontsize=10,
               fontweight='bold', bbox=dict(boxstyle='round,pad=0.4',
                                           facecolor='white', edgecolor='black'))
    
    # Add penalty annotations
    penalties = [
        (12.5, 0.5, 'Scaled Rewards\n+0.75 to +120', 'green'),
        (-12.5, 0.7, 'Penalty: -5\n(minimal)', 'darkgreen'),
        (-37.5, 0.5, 'Penalty: -5 to -50\n(linear escalation)', 'darkorange'),
        (-62.5, 0.3, 'Penalty: -50 to -150\n(heavy)', 'darkred'),
        (-87.5, 0.1, 'Penalty: -200\n(liquidation)', 'red'),
    ]
    
    for x, y, text, color in penalties:
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=color, linewidth=2))
    
    # Styling
    ax.set_xlim(-100, 300)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel('PNL (USDT)', fontsize=14, fontweight='bold')
    ax.set_title('Three-Tier Risk Zone Architecture\n10x Leverage | 100 USDT Position Size',
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('docs/20251024/risk_zones_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: docs/20251024/risk_zones_diagram.png")
    plt.show()


if __name__ == '__main__':
    print("Generating Three-Tier Risk Management Visualizations...\n")
    
    print("1. Exit Reward Function...")
    plot_exit_reward_function()
    
    print("\n2. Holding Feedback Function...")
    plot_holding_feedback()
    
    print("\n3. Reward Comparison Table...")
    plot_comparison_table()
    
    print("\n4. Risk Zones Diagram...")
    plot_risk_zones_diagram()
    
    print("\n✅ All visualizations generated successfully!")
    print("📁 Files saved in: docs/20251024/")
