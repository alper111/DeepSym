(define (domain test3)
  (:requirements :probabilistic-effects)
  (:predicates (a) (b) (c) (d) (e) (f) (g) (h))
  (:action a1
	   :effect (probabilistic 0.5 a
				  0.3 b
				  0.2 (and c (probabilistic 0.1 d
							    0.1 e
							    0.2 f
							    0.2 g
							    0.4 h)))))

(define (problem test3)
  (:domain test3))
