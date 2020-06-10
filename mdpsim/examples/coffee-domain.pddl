;; -*-lisp-*-
;; Coffee domain:
;;
;;   Dearden Richard, and Craig Boutilier.  1997.  Abstraction and
;;   approximate decision-theoretic planning.  Artificial
;;   Intelligence, 89(1-2):219-283.

(define (domain coffee)
  (:requirements :negative-preconditions :disjunctive-preconditions
		 :conditional-effects :probabilistic-effects :rewards)
  (:predicates (in-office) (raining) (has-umbrella) (is-wet)
	       (has-coffee) (user-has-coffee))
  (:action move
	   :effect (and (when (in-office)
			  (probabilistic 0.9 (not (in-office))))
			(when (not (in-office))
			  (probabilistic 0.9 (in-office)))
			(when (and (raining) (not (has-umbrella)))
			  (probabilistic 0.9 (is-wet)
					 0.1 (when (not (is-wet))
					       (increase (reward) 0.2))))
			(when (or (not (raining)) (has-umbrella))
			  (when (not (is-wet))
			    (increase (reward) 0.2)))
			(when (user-has-coffee)
			  (increase (reward) 0.8))))
  (:action buy-coffee
	   :effect (and (when (not (in-office))
			  (probabilistic 0.8 (has-coffee)))
			(when (user-has-coffee)
			  (increase (reward) 0.8))
			(when (not (is-wet))
			  (increase (reward) 0.2))))
  (:action get-umbrella
	   :effect (and (when (in-office)
			  (probabilistic 0.9 (has-umbrella)))
			(when (user-has-coffee)
			  (increase (reward) 0.8))
			(when (not (is-wet))
			  (increase (reward) 0.2))))
  (:action deliver-coffee
	   :effect (and (when (and (in-office) (has-coffee))
			  (probabilistic 0.8 (and (user-has-coffee)
						  (not (has-coffee))
						  (increase (reward) 0.8))
					 0.2 (and (probabilistic
						   0.5 (not (has-coffee)))
						  (when (user-has-coffee)
						    (increase (reward) 0.8)))))
			(when (and (not (in-office)) (has-coffee))
			  (and (probabilistic 0.8 (not (has-coffee)))
			       (when (user-has-coffee)
				 (increase (reward) 0.8))))
			(when (and (not (has-coffee)) (user-has-coffee))
			  (increase (reward) 0.8))
			(when (not (is-wet))
			  (increase (reward) 0.2)))))
