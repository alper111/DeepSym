;; Slippery gripper domain:
;;
;;   Kushmerick, Nicholas, Steve Hanks, and Daniel S. Weld.  1995.  An
;;   algorithm for probabilistic planning.  Artificial Intelligence,
;;   76(1-2):239-286.

(define (domain slippery-gripper)
  (:requirements :negative-preconditions :conditional-effects
		 :probabilistic-effects)
  (:predicates (gripper-dry) (holding-block))
  (:action pickup
	   :effect (and (when (gripper-dry)
			  (probabilistic 0.95 (holding-block)))
			(when (not (gripper-dry))
			  (probabilistic 0.5 (holding-block)))))
  (:action dry
	   :effect (probabilistic 0.8 (gripper-dry))))
