# Mobile Robot Kinematics

Following [this](https://web2.qatar.cmu.edu/~gdicaro/16311-Fall17/slides/siegvart-ch3.pdf) and [this](https://www.usna.edu/Users/cs/crabbe/SI475/current/mob-kin/mobkin.pdf) approaches, the main objective of this repo is to brush up some basic concepts of mobile robots.

This repo also contains some linear algebra tools

# Basic steps

This section describes the main steps to have a model working

## From input in the wheels, get a linear and angular velocity

The robot has active wheels, which means we are able to make them move as long as we provide voltage / current to it. The output of each wheel is a torque, that will lead to angular motion. Considering that there are fast controllers in each wheel, and that their control loop is way faster than our robot dynamics, we can consider that each will is controlled via __velocity setpoints__.

Using my first reference, we are able to calculate what is the angular and linear velocity that our robot is going to have at each step. The second reference is about to show how to use that information to compute where is going to be our next robot pose (`position in X, position in Y and Theta`)

## From our robot kinematics, compute next pose