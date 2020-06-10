(define (domain test3)
  (:requirements :probabilistic-effects)
  (:predicates (a) (b) (c) (d) (e) (f) (g) (h))
  (:action a1
	   :effect (probabilistic 0.5 a
				  0.3 b
				  0.2 c))
  (:action a2
	   :effect (probabilistic 0.9 a)))

(define (problem test3)
  (:domain test3))
