import pygame
import sys
import random
import math
from enum import Enum, auto
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ====================
# Simulation Parameters
# ====================
WIDTH, HEIGHT = 1000, 700
NUM_CYCLES = 15
AGV_COUNT = 3
FPS = 60

# Battery and Task Parameters
BATTERY_THRESHOLD = 50    # AGV can bid if battery > 50%
BATTERY_USAGE = 30        # Battery reduction when task completes (%)
CHARGE_RATE = 10           # Battery charging rate per second (%)
SPEED = 5                 # AGV movement speed (px/frame)
TASK_RADIUS = 20          # Task circle radius
WORKING_TIME = 2.0        # Task processing time (seconds)
TASK_GENERATION_INTERVAL = 5.0  # Time between new tasks (seconds)

# Colors
def rgb(r: int, g: int, b: int) -> Tuple[int, int, int]: 
    return (r, g, b)

WHITE = rgb(255, 255, 255)
BLACK = rgb(0, 0, 0)
GRAY = rgb(200, 200, 200)
RED = rgb(200, 0, 0)
GREEN = rgb(0, 200, 0)
BLUE = rgb(0, 0, 200)
YELLOW = rgb(255, 255, 0)
AGV_COLORS = [rgb(255, 165, 0), rgb(50, 205, 50), rgb(65, 105, 225)]
TASK_COLOR = RED
CHARGER_COLOR = GREEN
SHELF_COLOR = BLUE
PATH_COLOR = GRAY
TEXT_COLOR = BLACK

# Warehouse layout
CHARGER_POSITIONS = [
    (150, 150), (150, 300), (150, 450)
]
SHELF_POSITIONS = [
    (400, 150), (400, 300), (400, 450),
    (600, 150), (600, 300), (600, 450),
    (800, 150), (800, 300), (800, 450)
]
PATH_POINTS = [
    (150, 100), (850, 100), (850, 500), (150, 500), (150, 100)
]

class AGVState(Enum):
    IDLE = auto()
    TO_TASK = auto()
    WORKING = auto()
    TO_CHARGER = auto()
    CHARGING = auto()

class AGV:
    def __init__(self, idx: int):
        self.idx = idx
        self.pos = pygame.math.Vector2(CHARGER_POSITIONS[idx-1])
        self.battery = 100.0
        self.work_time = 0
        self.state = AGVState.IDLE
        self.target = None
        self.task_pos = None
        self.color = AGV_COLORS[idx-1]
        self.working_timer = 0.0
        self.path = []
        self.current_path_index = 0
        self.assigned_task_id = None
        self.battery_history = []
        self.utilization_time = 0.0
        self.idle_time = 0.0
        self.charging_time = 0.0
        self.state_history = []
        self.task_count = 0

    def can_bid(self) -> bool:
        return self.battery > BATTERY_THRESHOLD and self.state == AGVState.IDLE

    def needs_charging(self) -> bool:
        """Check if AGV needs charging (battery <= threshold)"""
        return self.battery <= BATTERY_THRESHOLD

    def bid_value(self) -> float:
        # Prioritize AGVs with lowest work_time and highest battery
        return self.work_time - (self.battery / 100)  # Lower is better

    def assign_task(self, task_pos: Tuple[int, int], task_id: int):
        self.state = AGVState.TO_TASK
        self.task_pos = pygame.math.Vector2(task_pos)
        self.target = self.task_pos
        self.assigned_task_id = task_id
        self.task_count += 1
        self.calculate_path()

    def calculate_path(self):
        """Improved path calculation with intermediate points"""
        self.path = []
        
        # Start from current position
        self.path.append(self.pos.copy())
        
        # Add intermediate points to follow warehouse path
        if abs(self.pos.x - self.target.x) > abs(self.pos.y - self.target.y):
            # Horizontal movement first
            self.path.append(pygame.math.Vector2(self.pos.x, self.target.y))
        else:
            # Vertical movement first
            self.path.append(pygame.math.Vector2(self.target.x, self.pos.y))
            
        # Add target position
        self.path.append(self.target.copy())
        
        self.current_path_index = 0

    def update(self, dt: float):
        # Record current state
        self.state_history.append((pygame.time.get_ticks(), self.state))
        
        # Track time spent in different states
        if self.state == AGVState.WORKING:
            self.utilization_time += dt
        elif self.state == AGVState.TO_TASK or self.state == AGVState.TO_CHARGER:
            self.utilization_time += dt * 0.5  # Moving is half utilization
        elif self.state == AGVState.CHARGING:
            self.charging_time += dt
        else:
            self.idle_time += dt
            
        # Record battery level for visualization
        self.battery_history.append((pygame.time.get_ticks(), self.battery))
        if len(self.battery_history) > 100:  # Keep last 100 records
            self.battery_history.pop(0)

        # Update position and state
        if self.state == AGVState.TO_TASK and self.target:
            self._move_along_path()
            if self._reached_target():
                self.state = AGVState.WORKING
                self.working_timer = 0.0
                
        elif self.state == AGVState.WORKING:
            self.working_timer += dt
            if self.working_timer >= WORKING_TIME:
                self.state = AGVState.TO_CHARGER
                self.target = pygame.math.Vector2(CHARGER_POSITIONS[self.idx-1])
                self.battery = max(0, self.battery - BATTERY_USAGE)
                self.assigned_task_id = None
                self.calculate_path()
                
        elif self.state == AGVState.TO_CHARGER and self.target:
            self._move_along_path()
            if self._reached_target():
                # FIXED: Only charge if battery needs charging (<=50%)
                if self.needs_charging():
                    self.state = AGVState.CHARGING
                else:
                    self.state = AGVState.IDLE
                self.target = None
                self.task_pos = None

        elif self.state == AGVState.CHARGING:
            self.battery = min(100.0, self.battery + CHARGE_RATE * dt)
            # FIXED: Stop charging when battery is full OR above threshold
            if self.battery >= 100.0:
                self.state = AGVState.IDLE

    def _move_along_path(self):
        """Move AGV along the calculated path"""
        if self.current_path_index < len(self.path):
            direction = (self.path[self.current_path_index] - self.pos)
            if direction.length() > 0:
                self.pos += direction.normalize() * SPEED
            # Move to next path point if close enough
            if self.pos.distance_to(self.path[self.current_path_index]) < SPEED:
                self.current_path_index += 1

    def _reached_target(self) -> bool:
        """Check if AGV has reached its target"""
        return self.current_path_index >= len(self.path)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        # Draw AGV body
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), 12)
        
        # Draw battery bar above
        self._draw_battery_bar(screen)
        
        # Draw AGV label
        self._draw_label(screen, font)
        
        # Draw work time indicator
        self._draw_work_time(screen, font)
        
        # Draw path if moving and has enough points
        if self.state in (AGVState.TO_TASK, AGVState.TO_CHARGER) and len(self.path) > 1:
            # Only draw if we have at least 2 points to connect
            points_to_draw = [(int(p.x), int(p.y)) for p in self.path[self.current_path_index:]]
            if len(points_to_draw) >= 2:
                pygame.draw.lines(screen, self.color, False, points_to_draw, 1)

    def _draw_battery_bar(self, screen: pygame.Surface):
        bar_w, bar_h = 30, 5
        x = self.pos.x - bar_w / 2
        y = self.pos.y - 20
        pygame.draw.rect(screen, BLACK, (x, y, bar_w, bar_h), 1)
        
        # Color battery bar based on level
        if self.battery > BATTERY_THRESHOLD:
            battery_color = GREEN
        elif self.battery > 20:
            battery_color = YELLOW
        else:
            battery_color = RED
            
        pygame.draw.rect(screen, battery_color, (x, y, bar_w * (self.battery / 100), bar_h))

    def _draw_label(self, screen: pygame.Surface, font: pygame.font.Font):
        state_text = {
            AGVState.IDLE: "IDL",
            AGVState.CHARGING: "CHG",
            AGVState.WORKING: "WRK",
            AGVState.TO_TASK: "→T",
            AGVState.TO_CHARGER: "→C"
        }.get(self.state, "")
        
        # Add battery percentage to label
        label = font.render(f"AGV{self.idx} {state_text} ({self.battery:.0f}%)", True, TEXT_COLOR)
        screen.blit(label, (self.pos.x - 35, self.pos.y + 15))

    def _draw_work_time(self, screen: pygame.Surface, font: pygame.font.Font):
        work_label = font.render(f"W:{self.work_time}", True, TEXT_COLOR)
        screen.blit(work_label, (self.pos.x - 10, self.pos.y + 30))

class Task:
    def __init__(self, task_id: int, position: Tuple[int, int]):
        self.task_id = task_id
        self.position = position
        self.creation_time = pygame.time.get_ticks()
        self.assignment_time = None
        self.completion_time = None
        self.assigned_to = None
        self.completed = False
        self.wait_times = []

def draw_warehouse(screen: pygame.Surface):
    """Draw the warehouse layout with shelves, chargers, and paths"""
    screen.fill(WHITE)
    
    # Draw main path
    pygame.draw.lines(screen, PATH_COLOR, False, PATH_POINTS, 3)
    
    # Draw charging stations
    for i, pos in enumerate(CHARGER_POSITIONS):
        pygame.draw.rect(screen, CHARGER_COLOR, (pos[0]-20, pos[1]-20, 40, 40))
        pygame.draw.rect(screen, BLACK, (pos[0]-20, pos[1]-20, 40, 40), 2)
        charger_label = pygame.font.SysFont(None, 20).render(f"CH {i+1}", True, BLACK)
        screen.blit(charger_label, (pos[0]-15, pos[1]-10))
    
    # Draw shelves
    for i, pos in enumerate(SHELF_POSITIONS):
        pygame.draw.rect(screen, SHELF_COLOR, (pos[0]-30, pos[1]-40, 60, 80))
        pygame.draw.rect(screen, BLACK, (pos[0]-30, pos[1]-40, 60, 80), 2)
        shelf_label = pygame.font.SysFont(None, 20).render(f"S{i+1}", True, BLACK)
        screen.blit(shelf_label, (pos[0]-5, pos[1]-15))

def draw_stats_panel(screen: pygame.Surface, font: pygame.font.Font, agvs: List[AGV], 
                    current_cycle: int, total_cycles: int, tasks: List[Task]):
    """Draw statistics panel showing simulation metrics"""
    panel_rect = pygame.Rect(WIDTH - 280, 20, 260, 220)
    pygame.draw.rect(screen, GRAY, panel_rect)
    pygame.draw.rect(screen, BLACK, panel_rect, 2)
    
    # Header
    title = font.render("Simulation Stats", True, BLACK)
    screen.blit(title, (panel_rect.x + 10, panel_rect.y + 10))
    
    # Cycle info
    cycle_text = f"Cycle: {current_cycle}/{total_cycles}"
    cycle_label = font.render(cycle_text, True, BLACK)
    screen.blit(cycle_label, (panel_rect.x + 10, panel_rect.y + 40))
    
    # Task info
    pending = sum(1 for t in tasks if not t.completed and not t.assigned_to)
    active = sum(1 for t in tasks if t.assigned_to and not t.completed)
    completed = sum(1 for t in tasks if t.completed)
    
    task_text = f"Tasks: {pending}P {active}A {completed}C"
    task_label = font.render(task_text, True, BLACK)
    screen.blit(task_label, (panel_rect.x + 10, panel_rect.y + 70))
    
    # AGV stats with improved formatting
    for i, agv in enumerate(agvs):
        total_time = agv.utilization_time + agv.idle_time + agv.charging_time
        utilization = (agv.utilization_time / total_time * 100) if total_time > 0 else 0
        
        # Status indicator
        status_color = {
            AGVState.IDLE: GRAY,
            AGVState.CHARGING: YELLOW,
            AGVState.WORKING: GREEN,
            AGVState.TO_TASK: BLUE,
            AGVState.TO_CHARGER: RED
        }.get(agv.state, GRAY)
        
        # Draw status circle
        status_x = panel_rect.x + 10
        status_y = panel_rect.y + 100 + i * 35
        pygame.draw.circle(screen, status_color, (status_x, status_y + 5), 5)
        
        agv_text = (f"AGV{i+1}: {utilization:.1f}% util "
                   f"{agv.battery:.0f}% bat")
        agv_label = font.render(agv_text, True, AGV_COLORS[i])
        screen.blit(agv_label, (status_x + 15, status_y))
        
        # Charging indicator
        if agv.needs_charging():
            charge_text = font.render("NEEDS CHARGE", True, RED)
            screen.blit(charge_text, (status_x + 15, status_y + 15))

def generate_visualizations(agvs: List[AGV], tasks: List[Task]):
    """Generate all required visualizations after simulation"""
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # 1. Bar chart: distribusi tugas per AGV
    plt.subplot(2, 2, 1)
    agv_ids = [f"AGV {i+1}" for i in range(len(agvs))]
    task_counts = [agv.task_count for agv in agvs]
    
    # Fixed color assignment - use list directly, not tuple wrapped in list
    colors = ['#FFA500', '#32CD32', '#4169E1'][:len(agvs)]
    
    bars = plt.bar(agv_ids, task_counts, color=colors)
    plt.title('Distribusi Tugas per AGV')
    plt.xlabel('AGV')
    plt.ylabel('Jumlah Tugas')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # 2. Line chart: level baterai tiap AGV dari waktu ke waktu
    plt.subplot(2, 2, 2)
    plot_colors = ['#FFA500', '#32CD32', '#4169E1']
    for i, agv in enumerate(agvs):
        if not agv.battery_history:
            continue
        times, batteries = zip(*agv.battery_history)
        times = [t/1000 for t in times]  # Convert to seconds
        plt.plot(times, batteries, label=f'AGV {agv.idx}', 
                color=plot_colors[i], linewidth=2)
    
    # Add battery threshold line
    plt.axhline(y=BATTERY_THRESHOLD, color='red', linestyle='--', 
                label=f'Threshold ({BATTERY_THRESHOLD}%)')
    
    plt.title('Level Baterai AGV')
    plt.xlabel('Waktu (detik)')
    plt.ylabel('Persentase Baterai (%)')
    plt.legend()
    plt.grid(True)
    
    # 3. Pie chart: proporsi waktu AGV dalam state IDLE, WORKING, CHARGING
    plt.subplot(2, 2, 3)
    # Calculate average proportions across all AGVs
    total_working = sum(agv.utilization_time for agv in agvs)
    total_idle = sum(agv.idle_time for agv in agvs)
    total_charging = sum(agv.charging_time for agv in agvs)
    total_time = total_working + total_idle + total_charging
    
    if total_time > 0:
        working_pct = total_working / total_time * 100
        idle_pct = total_idle / total_time * 100
        charging_pct = total_charging / total_time * 100
        
        sizes = [working_pct, idle_pct, charging_pct]
        labels = ['Working', 'Idle', 'Charging']
        colors = ['#FFA500', '#32CD32', '#4169E1']
        explode = (0.1, 0, 0)  # explode the working slice
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title('Proporsi Waktu Rata-rata Semua AGV')
    else:
        plt.text(0.5, 0.5, 'No time data', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 4. Wait time analysis for tasks
    plt.subplot(2, 2, 4)
    wait_times = []
    for task in tasks:
        if task.assignment_time:
            wait_time = (task.assignment_time - task.creation_time) / 1000  # in seconds
            wait_times.append(wait_time)
    
    if wait_times:
        plt.hist(wait_times, bins=10, color='#4169E1', edgecolor='black', alpha=0.7)
        plt.title('Distribusi Waktu Tunggu Tugas')
        plt.xlabel('Waktu Tunggu (detik)')
        plt.ylabel('Jumlah Tugas')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        avg_wait = np.mean(wait_times)
        max_wait = max(wait_times)
        plt.axvline(avg_wait, color='red', linestyle='--', label=f'Rata-rata: {avg_wait:.1f}s')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No wait time data', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('agv_simulation_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional metrics table
    print("\nMETRIK SIMULASI AGV")
    print("="*80)
    print(f"{'AGV':<10}{'Utilization (%)':<15}{'Idle Time (s)':<15}{'Charging Time (s)':<18}{'Tasks Completed':<15}")
    print("-"*80)
    for agv in agvs:
        total_time = agv.utilization_time + agv.idle_time + agv.charging_time
        utilization = agv.utilization_time / total_time * 100 if total_time > 0 else 0
        print(f"{f'AGV {agv.idx}':<10}{utilization:<15.1f}{agv.idle_time:<15.1f}{agv.charging_time:<18.1f}{agv.task_count:<15}")
    
    # Summary statistics
    print("\nRINGKASAN SIMULASI")
    print("-"*30)
    total_tasks = sum(agv.task_count for agv in agvs)
    completed_tasks = len([t for t in tasks if t.completed])
    avg_utilization = np.mean([agv.utilization_time / (agv.utilization_time + agv.idle_time + agv.charging_time) * 100 
                              for agv in agvs if (agv.utilization_time + agv.idle_time + agv.charging_time) > 0])
    
    print(f"Total tugas diselesaikan: {total_tasks}")
    print(f"Tugas yang selesai: {completed_tasks}")
    print(f"Rata-rata utilisasi AGV: {avg_utilization:.1f}%")
    
    if wait_times:
        print(f"Rata-rata waktu tunggu: {np.mean(wait_times):.1f} detik")
        print(f"Waktu tunggu maksimum: {max(wait_times):.1f} detik")

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AGV Warehouse Simulation - Fixed Charging Logic")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 20)
    small_font = pygame.font.SysFont('Arial', 16)

    # Initialize AGVs
    agvs = [AGV(i+1) for i in range(AGV_COUNT)]
    
    # Task management
    tasks = []
    last_task_time = 0
    task_counter = 0
    
    # Simulation control
    running = True
    paused = False
    current_cycle = 0
    simulation_time = 0.0

    while running:
        dt_ms = clock.tick(FPS)
        dt = dt_ms / 1000.0  # Convert to seconds
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:  # Press 'g' to generate visualizations
                    generate_visualizations(agvs, tasks)
        
        if paused:
            dt = 0  # Don't progress simulation when paused

        simulation_time += dt
        
        # Generate new tasks periodically
        if simulation_time - last_task_time > TASK_GENERATION_INTERVAL and len(tasks) < NUM_CYCLES:
            shelf_idx = random.randint(0, len(SHELF_POSITIONS)-1)
            task_pos = (SHELF_POSITIONS[shelf_idx][0], SHELF_POSITIONS[shelf_idx][1] - 20)
            tasks.append(Task(task_counter + 1, task_pos))
            task_counter += 1
            last_task_time = simulation_time

        # Draw warehouse
        draw_warehouse(screen)
        
        # Assign tasks to available AGVs
        for task in tasks:
            if not task.assigned_to and not task.completed:
                # Find AGV with best bid (lowest bid value)
                candidates = [agv for agv in agvs if agv.can_bid()]
                if candidates:
                    best_agv = min(candidates, key=lambda agv: agv.bid_value())
                    best_agv.assign_task(task.position, task.task_id)
                    task.assigned_to = best_agv.idx
                    task.assignment_time = pygame.time.get_ticks()
                    best_agv.work_time += 1
                    current_cycle += 1

        # Update AGVs
        for agv in agvs:
            agv.update(dt)
            agv.draw(screen, small_font)

        # Draw tasks
        for task in tasks:
            if task.completed:
                continue
                
            color = GREEN if task.assigned_to else RED
            pygame.draw.circle(screen, color, task.position, 8)
            task_label = small_font.render(f"T{task.task_id}", True, BLACK)
            screen.blit(task_label, (task.position[0] + 10, task.position[1] - 10))
            
            # Show waiting time for unassigned tasks
            if not task.assigned_to:
                wait_time = (pygame.time.get_ticks() - task.creation_time) / 1000
                wait_label = small_font.render(f"{wait_time:.1f}s", True, BLACK)
                screen.blit(wait_label, (task.position[0] - 20, task.position[1] + 15))

        # Mark completed tasks
        for agv in agvs:
            if agv.state == AGVState.TO_CHARGER and agv.assigned_task_id:
                for task in tasks:
                    if task.task_id == agv.assigned_task_id:
                        task.completed = True
                        task.completion_time = pygame.time.get_ticks()
                        break

        # Draw statistics panel
        draw_stats_panel(screen, small_font, agvs, current_cycle, NUM_CYCLES, tasks)
        
        # Draw pause indicator
        if paused:
            pause_text = font.render("PAUSED", True, RED)
            screen.blit(pause_text, (WIDTH // 2 - 40, 20))
        
        # Draw controls info
        controls_text = [
            "Controls:",
            "SPACE - Pause/Resume", 
            "G - Generate Charts",
            "ESC - Exit"
        ]
        for i, text in enumerate(controls_text):
            control_label = small_font.render(text, True, BLACK)
            screen.blit(control_label, (10, HEIGHT - 80 + i * 15))

        pygame.display.flip()

    # After simulation ends, generate visualizations
    generate_visualizations(agvs, tasks)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
