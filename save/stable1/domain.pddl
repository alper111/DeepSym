(define (domain stack)
	(:requirements :typing :negative-preconditions :probabilistic-effects :conditional-effects :disjunctive-preconditions)
	(:predicates
		(inserted) 
		(stacked) 
		(roll1) 
		(tumble1) 
		(roll2) 
		(tumble2) (base) 
		(pickloc ?x)
		(instack ?x)
		(stackloc ?x)
		(relation0 ?x ?y)
		(relation1 ?x ?y)
		(objtype0 ?x)
		(objtype1 ?x)
		(objtype2 ?x)
		(objtype3 ?x)
		(H0)
		(H1)
		(H2)
		(H3)
		(H4)
		(H5)
		(H6)
		(S0)
		(S1)
		(S2)
		(S3)
		(S4)
		(S5)
		(S6)
	)
	(:action stack0
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype0 ?below) (objtype0 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.014 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.113 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.070 (tumble1)
				 0.000 (roll2)
				 0.803 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack1
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype1 ?below) (objtype0 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.013 (roll1)
				 0.276 (tumble1)
				 0.000 (roll2)
				 0.711 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack2
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype0 ?below) (objtype1 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.048 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.048 (roll1)
				 0.032 (tumble1)
				 0.000 (roll2)
				 0.871 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack3
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype1 ?below) (objtype1 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.014 (roll1)
				 0.042 (tumble1)
				 0.000 (roll2)
				 0.944 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack4
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype0 ?below) (objtype0 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.448 (tumble1)
				 0.000 (roll2)
				 0.552 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack5
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype1 ?below) (objtype0 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.750 (tumble1)
				 0.000 (roll2)
				 0.250 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack6
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype0 ?below) (objtype1 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.421 (tumble1)
				 0.000 (roll2)
				 0.579 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack7
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype1 ?below) (objtype1 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.034 (roll1)
				 0.172 (tumble1)
				 0.000 (roll2)
				 0.793 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack8
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype0 ?below) (objtype2 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.048 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.055 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.630 (roll1)
				 0.068 (tumble1)
				 0.007 (roll2)
				 0.192 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack9
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype0 ?below) (objtype2 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.037 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.019 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.426 (roll1)
				 0.333 (tumble1)
				 0.000 (roll2)
				 0.185 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack10
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype1 ?below) (objtype2 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.024 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.634 (roll1)
				 0.238 (tumble1)
				 0.006 (roll2)
				 0.098 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack11
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype1 ?below) (objtype2 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.694 (roll1)
				 0.194 (tumble1)
				 0.000 (roll2)
				 0.111 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack12
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype0 ?below) (objtype3 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.071 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.012 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.131 (roll1)
				 0.012 (tumble1)
				 0.750 (roll2)
				 0.024 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack13
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype1 ?below) (objtype3 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.172 (roll1)
				 0.057 (tumble1)
				 0.770 (roll2)
				 0.000 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack14
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype0 ?below) (objtype3 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.062 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.125 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.000 (tumble1)
				 0.812 (roll2)
				 0.000 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack15
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype1 ?below) (objtype3 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.231 (tumble1)
				 0.769 (roll2)
				 0.000 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack16
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype2 ?below) (objtype0 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.006 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.229 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.006 (roll1)
				 0.017 (tumble1)
				 0.000 (roll2)
				 0.742 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack17
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype3 ?below) (objtype0 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.014 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.225 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.014 (tumble1)
				 0.000 (roll2)
				 0.746 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack18
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype2 ?below) (objtype0 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.381 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.048 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.571 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack19
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype3 ?below) (objtype0 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.586 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.310 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.103 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack20
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype2 ?below) (objtype1 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.024 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.751 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.053 (roll1)
				 0.000 (tumble1)
				 0.012 (roll2)
				 0.160 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack21
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype3 ?below) (objtype1 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.018 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.982 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.000 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack22
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype2 ?below) (objtype1 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.516 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.065 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.419 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack23
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype3 ?below) (objtype1 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.578 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.422 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.000 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack24
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype2 ?below) (objtype2 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.003 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.977 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.003 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.017 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack25
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype3 ?below) (objtype2 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.007 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.966 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.027 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack26
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype2 ?below) (objtype3 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.089 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.905 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.006 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack27
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype3 ?below) (objtype3 ?above) (relation0 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.984 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.016 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack28
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype2 ?below) (objtype2 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.000 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.950 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.050 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack29
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype2 ?below) (objtype3 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.312 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.688 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.000 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack30
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype3 ?below) (objtype2 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.750 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.154 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.096 (tumble2))
				(not (pickloc ?above)))
	)
	(:action stack31
		:parameters (?below ?above)
		:precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) (objtype3 ?below) (objtype3 ?above) (relation1 ?below ?above))
		:effect (and (probabilistic
				 0.605 (and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.237 (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))
				 0.000 (roll1)
				 0.000 (tumble1)
				 0.000 (roll2)
				 0.158 (tumble2))
				(not (pickloc ?above)))
	)
	(:action increase-height1
		:precondition (and (stacked) (H0))
		:effect (and (not (H0)) (H1) (not (stacked)))
	)
	(:action increase-height2
		:precondition (and (stacked) (H1))
		:effect (and (not (H1)) (H2) (not (stacked)))
	)
	(:action increase-height3
		:precondition (and (stacked) (H2))
		:effect (and (not (H2)) (H3) (not (stacked)))
	)
	(:action increase-height4
		:precondition (and (stacked) (H3))
		:effect (and (not (H3)) (H4) (not (stacked)))
	)
	(:action increase-height5
		:precondition (and (stacked) (H4))
		:effect (and (not (H4)) (H5) (not (stacked)))
	)
	(:action increase-height6
		:precondition (and (stacked) (H5))
		:effect (and (not (H5)) (H6) (not (stacked)))
	)
	(:action increase-stack1
		:precondition (and (inserted) (S0))
		:effect (and (not (S0)) (S1) (not (inserted)))
	)
	(:action increase-stack2
		:precondition (and (inserted) (S1))
		:effect (and (not (S1)) (S2) (not (inserted)))
	)
	(:action increase-stack3
		:precondition (and (inserted) (S2))
		:effect (and (not (S2)) (S3) (not (inserted)))
	)
	(:action increase-stack4
		:precondition (and (inserted) (S3))
		:effect (and (not (S3)) (S4) (not (inserted)))
	)
	(:action increase-stack5
		:precondition (and (inserted) (S4))
		:effect (and (not (S4)) (S5) (not (inserted)))
	)
	(:action increase-stack6
		:precondition (and (inserted) (S5))
		:effect (and (not (S5)) (S6) (not (inserted)))
	)
	(:action makebase
		:parameters (?obj)
		:precondition (not (base))
		:effect (and (base) (stacked) (inserted) (not (pickloc ?obj)) (stackloc ?obj))
	)
)
