(define (problem slippery-gripper)
  (:domain slippery-gripper)
  (:init (probabilistic 0.7 (gripper-dry)))
  (:goal (holding-block)))
