(define (problem dom1) (:domain stack)
	(:objects O1 O2 O3 O4)
	(:init
		(pickloc O1) (objtype3 O1)
		(pickloc O2) (objtype3 O2)
		(pickloc O3) (objtype3 O3)
		(pickloc O4) (objtype3 O4)
		(relation1 O1 O2)
		(relation1 O1 O3)
		(relation1 O1 O4)
		(relation0 O2 O1)
		(relation1 O2 O3)
		(relation1 O2 O4)
		(relation0 O3 O1)
		(relation0 O3 O2)
		(relation1 O3 O4)
		(relation0 O4 O1)
		(relation0 O4 O2)
		(relation0 O4 O3)
		(H0)
		(S0)
	)
	(:goal (and (H3) (S4) (not (stacked)) (not (inserted))))
)
