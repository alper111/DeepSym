;; -*-lisp-*-
(define (domain simple)
  (:requirements :negative-preconditions :conditional-effects :mdp)
  (:predicates (p1) (p2) (p3))
  (:action a1
	   ;; This effect has inconsistent rewards
	   :effect (probabilistic 0.5 (increase (reward) 1))))
	   ;; This effect is inconsistent
;	   :effect (and (p1) (not (p1)))))
;	   :effect (and (when (p1) (and (not (p1)) (increase (reward) 1)))
;			(when (p2) (and (not (p2)) (increase (reward) 2)))
;			(when (p1) (and (not (p1)) (increase (reward) 4))))))
;	   :effect (and (when (not (p1)) (probabilistic 0.2 (p1)))
;			(probabilistic 0.5 (when (not (p2)) (p2))))))
;	   :effect (probabilistic 0.5 (when (not (p2))
;					(probabilistic 0.2 (when (not (p1))
;							     (p1))
;						       0.6 (p2)))
;				  0.2 (not (p2)))))

(define (problem simple)
  (:domain simple))
