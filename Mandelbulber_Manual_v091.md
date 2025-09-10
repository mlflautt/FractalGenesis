    Query successful

Here is the information from the user-provided PDF converted to a Markdown file.

Mandelbulber: Tutorial & User Manual

Version: 0.9.1 (22nd, September 2016) based on Mandelbulber Version 2.0.9

    Download: https://sourceforge.net/projects/mandelbulber/ 

Project Website: https://github.com/buddhi1980/mandelbulber2 

Forum: http://www.fractalforums.com/mandelbulber/ 

Editors: Krzysztof Marczak, Graeme McLarekin, Sebastian Jennen 

Table of Contents

    What are fractals? 


1.1. Mandelbrot set 


1.2. 3D fractals 


1.3. Mandelbulber Program 

Distance Estimation 

Ray-marching - Maximum number of iterations vs. distance threshold condition 

Navigation 


4.1. Camera and Target movement step 


4.2. The camera and target functions 


4.3. Linear camera and target movement modes using the arrow buttons 


4.4. Linear camera and target movement modes using the mouse pointer 


4.5. Camera rotation modes using the arrow buttons 


4.6. Reset View 


4.7. Calculation of rotation angles modes 


4.8. Camera rotation in animations 

Interpolation 


5.1. Akima interpolation 


5.2. Catmul-Rom / Akima interpolation - Advices 


5.3. Changing interpolation types 

NetRender 


6.1. Starting NetRender 

Q&A, Examples and Hints 


7.1. Q&A. How do you get different materials on different shapes? 

What are fractals?

Fractals are objects that have 

self-similarity, where smaller fragments are similar to those on a larger scale. A key feature is that they contain subtle details even at very high magnification.

Mandelbrot set

The Mandelbrot set is a typical example of a two-dimensional fractal generated mathematically. The image is created with a simple formula calculated over many iterations: 

zn+1​=zn2​+c 

In this formula:

    z is a complex number (a+ib), where 'a' is the real part and 'ib' is the imaginary part.

c represents the coordinates of the image point being iterated.

Initially, the value of 'z' is set to 'c' (

z0=c), and this parameter is then used repeatedly in the iteration loop.

Each original point (pixel) is tested in an iteration loop to determine if it belongs to the fractal set.

Termination Conditions
To prevent the formula from iterating indefinitely, termination conditions are used, most commonly 

Bailout and Maxiter.

    Maxiter is a condition that stops the iteration when a maximum number of iterations is reached.

Bailout stops the iteration if the point moves further than a set distance from the origin. For the Mandelbrot formula, the modulus (length of the vector from the origin to the current 'z' point, also called 'r') is calculated after each iteration. If 

r>2 (Bailout = 2), the iteration stops and the point is marked with a light color. If 'r' remains less than 2 after many iterations, it's considered for simplicity to continue indefinitely, and the iteration is stopped at Maxiter, with the point marked in black. This creates the black "set" of points that don't reach bailout, while others are given lighter colors.

3D fractals

The three-dimensional fractal, the "Mandelbulb," is calculated similarly to the Mandelbrot set, but the vector "z" contains three complex numbers (x, y, z) or four dimensions (x, y, z, w). They are denoted as (z.x, z.y, z.z). Other 3D fractals, like the Menger Sponge and Sierpinski pyramid, are based on Iterated Function Systems (IFS).

Mandelbulber Program

Mandelbulber is a user-friendly application for rendering 3D fractals, including the Mandelbulb, Mandelbox, and Menger Sponge.

Distance Estimation

Distance Estimation (DE) is an algorithm that calculates an approximate distance from a given point to the nearest surface of the fractal. It is the most crucial algorithm for rendering 3D fractals efficiently.

Without DE, tracing a "photon" (a simulated beam of light from the camera) toward the fractal surface would require many small steps, possibly up to 10,000 per pixel. DE allows the step size along the ray to be increased based on the distance estimate, a process known as 

ray-marching.

The ray-marching process moves the photon along the ray by the estimated distance, with the distance becoming smaller as it gets closer to the fractal surface, making the process more accurate. Ray-marching stops when the photon is within a set 

distance threshold of the surface or after a maximum number of iterations if that option is enabled.

Sometimes the distance estimation can be inaccurate, leading to the photon "overshooting" and passing through the fractal surface, which can cause noise in the image. To mitigate this, you can use a 

"ray-marching step multiplier" (a number between 0 and 1) to make the steps smaller, reducing the risk of overshooting but increasing render time.

Mandelbulber uses two main DE modes:

    Analytical DE: Faster to calculate and is the preferred mode for most formulas.

Delta DE: Produces a good-quality image for some formulas where Analytical DE doesn't.

Both DE modes can be used with either 

linear or logarithmic DE functions. The best combination of DE mode and function, as well as formula parameters, is specific to the fractal.

The 

Statistics tab shows the Percentage of Wrong Distance Estimations ("Bad DE"). As a general guideline, a value less than 0.01 is good, though this can vary by case.

Ray-marching - Maximum number of iterations vs. distance threshold condition

The 

ray-marching distance threshold is the distance at which ray-marching stops. It controls the level of detail in the image and can be set to vary so that closer surfaces have greater detail.

Ray-marching can be stopped in two ways:

    Stop at distance threshold (when "Stop at maximum iteration" is disabled). Ray-marching continues until the photon reaches the specified distance threshold from the fractal surface.

Stop at maximum number of iterations (when "Stop at maximum iteration" is enabled). Ray-marching stops when the photon has taken the maximum number of steps, regardless of whether it has reached the distance threshold.

It's important to note that the "Stop at maximum iteration" setting for ray-marching is separate from the fractal iteration loop's Maxiter condition. The fractal iteration loop still runs to achieve Bailout or Maxiter, regardless of the ray-marching setting.

Navigation

To navigate the scene in Mandelbulber, you control two main elements:

    Camera: The position from which you view the fractal.

Target: The point the camera is always looking at.

Camera and Target movement step

Movement can be controlled by changing values in the spin boxes or using "steps". The user can define a rotation step (default 15 degrees) and a linear movement step (default 0.5).

There are two modes for linear movement steps:

    Relative step mode: The step size is calculated based on the estimated distance from the camera to the fractal surface. The closer the camera is, the smaller the step, which prevents the camera from moving through the fractal. This mode is recommended for animations that approach the fractal's surface.

Absolute step mode: The step size is fixed. This is recommended for animations where the camera moves at a fixed or controlled speed.

Linear camera and target movement modes using the arrow buttons

Three modes are available for linear movement using the arrow keys on the Navigation dock:

    Move camera and target mode: Both the camera and target move the same distance in the same direction, with the camera's angle of rotation remaining unchanged. 2.  

Move camera mode: Only the camera moves, and it rotates to keep the stationary target in view. 3.  

Move target mode: Only the target moves, and the fixed camera rotates to follow it. This mode will not work in Relative Step Mode if the target is inside the fractal (distance = 0).

Camera rotation modes using the arrow buttons

Two rotation modes are available:

    Rotate camera: Rotates the camera around its own axis, moving the target accordingly. This is the standard rotation mode. 2.  

Rotate around target: The camera moves around the stationary target while maintaining a constant distance, and it rotates to continue looking at the target.

Reset View

The 

Reset View function zooms the camera out from the fractal while preserving the camera's angles. If you set the rotation angles to zero before using it, the camera will zoom out and be rotated to look down the y-axis.

Calculation of rotation angles modes

Two modes are available for calculating rotation angles:

    Fixed-roll angle: The 'gamma' (roll) angle remains constant. This is likened to aircraft controls, where turns are relative to the aircraft's axis.

Straight rotation: The camera rotates around its own local axis, which is more intuitive. This mode changes the gamma (roll) angle automatically.

Camera rotation in animations

In animations, the camera and target can move independently, with the camera always looking at the target. The camera's rotation angle is determined by the coordinates of both the camera and the target.

Sometimes, uneven movement between keyframes can lead to unexpected camera rotations. To fix this, you can use the "Set the same distance from the camera for all the frames" button on the Keyframe navigation tab. This sets a constant distance between the camera and target for all keyframes without changing the visual effect, which helps to correct interpolation.

Interpolation

Mandelbulber can use 

Catmull-Rom and Akima interpolation to create smooth transitions for values between keyframes in animations. Akima interpolation stays closer to the sample points, while Catmull-Rom oscillates a bit more.

Collision advice
When approaching an obstacle in an animation, a fast approach may cause the camera to be inadvertently dragged towards the object's centerIt is recommended that the distance to the object does not decrease by more than 5 times between keyframes

NetRender

NetRender is a tool that allows you to render a single image or animation across multiple computers simultaneously over an Ethernet network, significantly increasing computing power.

    One computer acts as the 

    Server (master), managing the rendering process, sending requests to clients, and collecting the results. The server also renders a portion of the image and combines the results.

The other computers act as 

Clients (slaves), rendering different portions of the image and sending them to the server.

The total number of CPU cores used is the sum of the server's cores and all connected clients' cores.

Starting NetRender

    Server configuration: On the server computer, set the mode to Server and ensure the local server port is not in use and is open through any firewalls. The default port is 5555. Then, press the "Launch server and watch for clients" button.

Client configuration: On the client computers, set the mode to Client. Enter the server computer's IP address or name in the "Remote server address" field and the correct port number in the "Remote server port" field. Then, press "Connect to server". A "READY" status indicates a successful connection.

Rendering: Only the server can initiate rendering. When the RENDER button is pressed on the server, all connected computers start rendering their assigned portions.

Q&A, Examples and Hints

    Hybrid Fractals: When creating a hybrid fractal, using a single iteration of a second fractal is often the best practice.

Wrong Distance Estimations: Monitoring the "Percentage of Wrong Distance Estimations" is a good way to manage image quality. A value of less than 0.01 is generally good, but this can vary.

Raymarching Step Multiplier: The ray-marching step multiplier (also called "fudge factor") can be adjusted to balance quality and render time. In animations, it may be necessary to lower it to account for changes between keyframes.

MandelboxMenger Hybrids: The percentage of Bad DE in MandelboxMenger Hybrids often improves as you zoom in.

Maximum View Distance: This setting should be optimized to reduce render time. You can lower it until the furthest parts of the object start to disappear, but allow for changes in animation. Be aware that a mouse click in Relative Step Mode on 

spherical_inversion can reset this value to 280, increasing render times if not adjusted.

Magic Angle Benesi Mag Transforms: To render these fractals parallel to the x-y plane, set the y-axis rotation on the fractal dock to 35.2644∘ (=90∘−54.7356∘).

Transform_Menger Fold: This transform can be split into a start and end function and used in Hybrid Mode.

Hybrid Iteration Controls: You can use iteration controls to tweak hybrids. For example, you can set a slot to repeat for a certain number of iterations before moving to the next slot.

Q&A. How do you get different materials on different shapes?

The document explains one way to get different materials on different shapes.

    Material Manager (A): This is where you can start a new material or load an existing one. The active material is highlighted in blue and is active in the material editor for creation or modification.

Material Selection (B): To apply a material, go to Global Parameters and click on the material preview image. A Material Manager UI will appear, showing the materials you have loaded or created. Click on the one you want to use and close the UI. You can also click on the material preview image for primitives or for each fractal/transform in Boolean Mode.

