from manim import *
import numpy as np
import torch

class PointProgression(Scene):
    def construct(self):
        # Configuration
        torch.manual_seed(42)  # Match the seed from the original code
        np.random.seed(42)
        num_points = 1000
        num_steps = 10  # Number of steps to visualize in the animation
        
        # Create simulated data similar to your diffusion process
        # Initial random points (xt)
        xt = torch.randn(num_points, 2).numpy()
        
        # Target points (sampled_points) - let's create some interesting pattern
        theta = np.linspace(0, 2*np.pi, num_points//2)
        radius_inner = 2
        radius_outer = 3
        
        # Create a double-ring pattern
        sampled_points = np.zeros((num_points, 2))
        sampled_points[:num_points//2, 0] = radius_inner * np.cos(theta)
        sampled_points[:num_points//2, 1] = radius_inner * np.sin(theta)
        sampled_points[num_points//2:, 0] = radius_outer * np.cos(theta)
        sampled_points[num_points//2:, 1] = radius_outer * np.sin(theta)
        
        # Add some noise to make it look more natural
        sampled_points += np.random.randn(*sampled_points.shape) * 0.1
        
        # Simulate the diffusion process steps
        # In a real model, this would be predictions, but here we'll just 
        # interpolate between the initial random points and the target distribution
        steps = []
        for t_val in np.linspace(0, 1, num_steps):
            # Simple linear interpolation - in a real diffusion model this would be model output
            current_points = xt * (1 - t_val) + sampled_points * t_val
            steps.append(current_points)
        
        # Title and axis
        title = Text("Diffusion Model Sampling Process", font_size=36)
        title.to_edge(UP)
        
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            axis_config={"include_tip": False}
        )
        
        # Create the dot groups - one for target points (sampled_points) and one for evolving points (xt)
        target_dots = VGroup(*[Dot(point=np.array([p[0], p[1], 0]), color=BLUE, radius=0.03) 
                             for p in sampled_points])
        
        evolving_dots = VGroup(*[Dot(point=np.array([p[0], p[1], 0]), color=GREEN, radius=0.03) 
                               for p in xt])
        
        # Labels
        step_label = Text("Step: 0", font_size=24)
        step_label.to_edge(DOWN).shift(UP * 0.5)
        
        # Animation sequence
        self.play(Write(title))
        self.play(Create(axes))
        
        # Show target distribution first
        self.play(Create(target_dots, lag_ratio=0.01, run_time=1.5))
        target_label = Text("Target Distribution", color=BLUE, font_size=24)
        target_label.next_to(axes, DOWN)
        self.play(Write(target_label))
        self.wait(1)
        
        # Show initial random points
        self.play(Create(evolving_dots, lag_ratio=0.01, run_time=1.5))
        initial_label = Text("Initial Random Points", color=GREEN, font_size=24)
        initial_label.next_to(target_label, DOWN)
        self.play(Write(initial_label))
        self.wait(1)
        
        # Now animate the progression
        self.add(step_label)
        
        for i, step_points in enumerate(steps):
            # Update the step label
            new_step_label = Text(f"Step: {i+1}/{num_steps}", font_size=24)
            new_step_label.to_edge(DOWN).shift(UP * 0.5)
            
            # Calculate color interpolation from GREEN to YELLOW
            t = i / (num_steps - 1)
            current_color = interpolate_color(GREEN, YELLOW, t)
            
            # Create the animation for moving dots and changing color
            animations = []
            for j, dot in enumerate(evolving_dots):
                new_position = np.array([step_points[j, 0], step_points[j, 1], 0])
                animations.append(dot.animate.move_to(new_position).set_color(current_color))
            
            # Play the animations for this step
            self.play(
                *animations,
                Transform(step_label, new_step_label),
                run_time=0.8
            )
        
        # Add final label
        final_label = Text("Final Distribution", color=YELLOW, font_size=24)
        final_label.next_to(initial_label, DOWN)
        self.play(Write(final_label))
        
        self.wait(2)


# Enhanced version that shows progress more effectively
class OptimizedPointProgression(Scene):
    def construct(self):
        # Configuration
        torch.manual_seed(42)
        np.random.seed(42)
        num_points = 1000
        num_steps = 20  # More steps for smoother animation
        
        # Create simulated data
        xt = torch.randn(num_points, 2).numpy() * 3  # Start with wider distribution
        
        # Generate target distribution - two concentric circles with some noise
        theta = np.linspace(0, 2*np.pi, num_points//2, endpoint=False)
        radius_inner = 2
        radius_outer = 3.5
        
        sampled_points = np.zeros((num_points, 2))
        # Inner circle
        sampled_points[:num_points//2, 0] = radius_inner * np.cos(theta)
        sampled_points[:num_points//2, 1] = radius_inner * np.sin(theta)
        # Outer circle
        sampled_points[num_points//2:, 0] = radius_outer * np.cos(theta)
        sampled_points[num_points//2:, 1] = radius_outer * np.sin(theta)
        
        # Add some noise to make it look more realistic
        sampled_points += np.random.randn(*sampled_points.shape) * 0.15
        
        # Performance optimization: Use Point cloud instead of individual dots
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            axis_config={"include_tip": False},
        ).set_opacity(0.3)
        
        target_points = [np.array([p[0], p[1], 0]) for p in sampled_points]
        evolving_points = [np.array([p[0], p[1], 0]) for p in xt]
        
        target_cloud = PMobject()
        target_cloud.add_points(target_points)
        target_cloud.set_color(BLUE)
        
        evolving_cloud = PMobject()
        evolving_cloud.add_points(evolving_points)
        evolving_cloud.set_color(GREEN)
        
        # Title and information display
        title = Text("Diffusion Process Visualization", font_size=36)
        title.to_edge(UP)
        
        # Step counter and progress bar
        step_text = Text("Step: 0/20", font_size=24)
        step_text.to_corner(DR)
        
        progress_bar_bg = Rectangle(height=0.2, width=4, color=WHITE, fill_opacity=0.2)
        progress_bar_bg.to_corner(DL).shift(RIGHT * 2 + UP * 0.5)
        
        progress_bar = Rectangle(height=0.2, width=0, color=YELLOW, fill_opacity=1)
        progress_bar.align_to(progress_bar_bg, LEFT)
        progress_bar.to_corner(DL).shift(RIGHT * 2 + UP * 0.5)
        
        # Animation sequence
        self.play(Write(title), Create(axes))
        self.play(Create(target_cloud, lag_ratio=0.01, run_time=1))
        
        target_label = Text("Target Distribution", color=BLUE, font_size=20)
        target_label.to_edge(LEFT).shift(UP * 2)
        self.play(Write(target_label))
        
        self.play(Create(evolving_cloud, lag_ratio=0.01, run_time=1))
        
        evolving_label = Text("Evolving Points", color=GREEN, font_size=20)
        evolving_label.next_to(target_label, DOWN)
        self.play(Write(evolving_label))
        
        # Add progress indicators
        self.add(step_text, progress_bar_bg, progress_bar)
        
        # Simulate the diffusion process
        for i in range(num_steps):
            # Calculate interpolation factor with easing for more realistic diffusion
            t = (i+1) / num_steps
            # Apply ease_out_cubic for more natural diffusion effect
            eased_t = 1 - (1 - t) ** 3
            
            # Smoothly interpolate positions with a bit of noise to simulate stochasticity
            new_points = np.array([(1 - eased_t) * xt[j] + eased_t * sampled_points[j] for j in range(num_points)])
            
            # Add decreasing noise to simulate the denoising process
            noise_scale = 0.5 * (1 - t)
            new_points += np.random.randn(*new_points.shape) * noise_scale
            
            # Convert to Manim points
            new_manim_points = [np.array([p[0], p[1], 0]) for p in new_points]
            
            # Create a new point cloud for this step
            new_cloud = PMobject()
            new_cloud.add_points(new_manim_points)
            
            # Interpolate color: GREEN -> YELLOW
            color = interpolate_color(GREEN, YELLOW, eased_t)
            new_cloud.set_color(color)
            
            # Update step text and progress bar
            new_step_text = Text(f"Step: {i+1}/{num_steps}", font_size=24)
            new_step_text.to_corner(DR)
            
            new_progress_bar = Rectangle(
                height=0.2, 
                width=4 * (i+1)/num_steps, 
                color=YELLOW, 
                fill_opacity=1
            )
            new_progress_bar.align_to(progress_bar_bg, LEFT)
            new_progress_bar.to_corner(DL).shift(RIGHT * 2 + UP * 0.5)
            
            # Create noise visualization
            noise_cloud = PMobject()
            noise_points = [np.array([p[0], p[1], 0]) for p in np.random.randn(num_points, 2) * noise_scale * 3]
            noise_cloud.add_points(noise_points)
            noise_cloud.set_color(TEAL)
            
            
            # Now play the main animation
            self.play(
                Transform(evolving_cloud, new_cloud),
                Transform(evolving_label, Text(f"Evolving Points (t={eased_t:.2f})", color=color, font_size=20).next_to(target_label, DOWN)),
                Transform(step_text, new_step_text),
                Transform(progress_bar, new_progress_bar),
                run_time=1
            )
        
        # Final state
        final_label = Text("Final Distribution", color=YELLOW, font_size=20)
        final_label.next_to(evolving_label, DOWN)
        self.play(Write(final_label))
        
        self.wait(2)

# run this animation:
# manim -pql main.py OptimizedPointProgression
# for higher quality:
# manim -pqh main.py OptimizedPointProgression