(define (domain tiger-domain)
  (:requirements :negative-preconditions :conditional-effects
		 :probabilistic-effects :rewards)
  (:predicates (tiger-on-left) (hear-tiger-on-left))
  (:action listen
	   :effect (and (when (tiger-on-left)
			  (probabilistic 0.85 (hear-tiger-on-left)
					 0.15 (not (hear-tiger-on-left))))
			(when (not (tiger-on-left))
			  (probabilistic 0.85 (not (hear-tiger-on-left))
					 0.15 (hear-tiger-on-left)))))
  (:action open-left-door
	   :effect (and (when (not (tiger-on-left))
			  (increase (reward) 100))
			(when (tiger-on-left)
			  (decrease (reward) 100))))
  (:action open-right-door
	   :effect (and (when (tiger-on-left)
			  (increase (reward) 100))
			(when (not (tiger-on-left))
			  (decrease (reward) 100)))))

(define (problem tiger-problem)
  (:domain tiger-domain)
  (:init (probabilistic 0.5 (tiger-on-left)))
  (:metric maximize (reward)))
