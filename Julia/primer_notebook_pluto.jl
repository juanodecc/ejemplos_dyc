### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 245de79e-260c-11ec-082f-bdef59a44c40
using ControlSystems


# ╔═╡ f5e08ada-5f12-45a2-ba56-62af6bbe0a0d
begin
	J = 2.0
	b = 0.04
	K = 1.0
	R = 0.08
	L = 1e-4
	
	# dc motor model
	s = tf("s")
	P = K/(s*((J*s + b)*(L*s + R) + K^2))
	
	# Create an array of closed loop systems for different values of kp
	Tcl = TransferFunction[kp*P/(1 + kp * P) for kp = [1, 5, 15]]
	
	# Closed loop step response
	stepplot(Tcl, label= ["kp = 1" "kp = 5" "kp = 15"])
end

# ╔═╡ 3f85cf01-bcd7-4395-84bf-2c5eb597f497


# ╔═╡ Cell order:
# ╠═245de79e-260c-11ec-082f-bdef59a44c40
# ╠═f5e08ada-5f12-45a2-ba56-62af6bbe0a0d
# ╠═3f85cf01-bcd7-4395-84bf-2c5eb597f497
