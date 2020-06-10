;; Extended slippery gripper domain:
;;
;;   Kushmerick, Nicholas, Steve Hanks, and Daniel S. Weld.  1995.  An
;;   algorithm for probabilistic planning.  Artificial Intelligence,
;;   76(1-2):239-286.

(define (domain ext-slippery-gripper)
  (:requirements :negative-preconditions :conditional-effects
		 :probabilistic-effects)
  (:predicates (gripper-dry) (holding-block) (block-painted) (gripper-clean))
  (:action pickup
	   :effect (and (when (gripper-dry)
			  (probabilistic 0.95 (holding-block)))
			(when (not (gripper-dry))
			  (probabilistic 0.5 (holding-block)))))
  (:action dry
	   :effect (probabilistic 0.8 (gripper-dry)))
  (:action paint
	   :effect (and (block-painted)
			(when (not (holding-block))
			  (probabilistic 0.1 (not (gripper-clean))))
			(when (holding-block)
			  (not (gripper-clean))))))
