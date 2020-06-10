(define (domain test2)
  (:requirements :rewards)
  (:predicates (a) (b))
  (:action a1
	   :precondition (a)
	   :effect (and (b) (decrease (reward) 1))))

(define (problem test2)
  (:domain test2)
  (:requirements :probabilistic-effects)
  (:init (probabilistic 1/2 (a) 1/2 (b)))
  (:goal (b))
  (:goal-reward 100)
  (:metric maximize (reward)))
