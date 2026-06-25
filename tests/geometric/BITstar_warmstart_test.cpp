/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2026, Metal Shark Boats
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: STcore */

#define BOOST_TEST_MODULE "BITstarWarmStart"
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <vector>

#include "ompl/base/PlannerData.h"
#include "ompl/base/ProblemDefinition.h"
#include "ompl/base/SpaceInformation.h"
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"
#include "ompl/base/spaces/RealVectorStateSpace.h"
#include "ompl/base/terminationconditions/IterationTerminationCondition.h"
#include "ompl/geometric/planners/informedtrees/BITstar.h"
#include "ompl/util/RandomNumbers.h"

using namespace ompl;

namespace
{
    // Axis-aligned obstacle box. Any state inside [LOW, HIGH] on both axes is invalid; everything
    // else in the bounded space is free. A straight motion that passes through the box is rejected
    // by the discrete motion validator, which lets us build both feasible and infeasible edges.
    constexpr double OBSTACLE_LOW = 4.0;
    constexpr double OBSTACLE_HIGH = 6.0;
    constexpr double SPACE_MIN = 0.0;
    constexpr double SPACE_MAX = 10.0;

    bool isStateValid(const base::State *state)
    {
        const auto *xy = state->as<base::RealVectorStateSpace::StateType>();
        const double x = xy->values[0];
        const double y = xy->values[1];
        const bool insideBox =
            (x >= OBSTACLE_LOW && x <= OBSTACLE_HIGH && y >= OBSTACLE_LOW && y <= OBSTACLE_HIGH);
        return !insideBox;
    }

    /// \brief A 2D planning problem with the obstacle box and a BITstar planner ready to solve.
    /// The start and goal are caller-supplied; the warm-start path is set separately.
    struct WarmStartFixture
    {
        base::StateSpacePtr space;
        base::SpaceInformationPtr si;
        base::ProblemDefinitionPtr pdef;
        std::shared_ptr<geometric::BITstar> planner;

        WarmStartFixture(const std::vector<double> &start, const std::vector<double> &goal)
        {
            auto rvSpace = std::make_shared<base::RealVectorStateSpace>(2);
            base::RealVectorBounds bounds(2);
            bounds.setLow(SPACE_MIN);
            bounds.setHigh(SPACE_MAX);
            rvSpace->setBounds(bounds);
            space = rvSpace;

            si = std::make_shared<base::SpaceInformation>(space);
            si->setStateValidityChecker(isStateValid);
            // Fine resolution so a straight edge crossing the box is reliably caught as infeasible.
            si->setStateValidityCheckingResolution(0.005);
            si->setup();

            pdef = std::make_shared<base::ProblemDefinition>(si);

            base::ScopedState<base::RealVectorStateSpace> startState(space);
            startState[0] = start[0];
            startState[1] = start[1];
            base::ScopedState<base::RealVectorStateSpace> goalState(space);
            goalState[0] = goal[0];
            goalState[1] = goal[1];
            pdef->setStartAndGoalStates(startState, goalState);
            pdef->setOptimizationObjective(
                std::make_shared<base::PathLengthOptimizationObjective>(si));

            planner = std::make_shared<geometric::BITstar>(si);
            planner->setProblemDefinition(pdef);
            planner->setup();
        }

        /// \brief Build a warm-start path from a list of (x, y) waypoints and hand it to the planner.
        void setWarmStartPath(const std::vector<std::vector<double>> &waypoints)
        {
            std::vector<base::State *> owned;
            owned.reserve(waypoints.size());
            std::vector<const base::State *> path;
            path.reserve(waypoints.size());
            for (const auto &wp : waypoints)
            {
                base::State *s = si->allocState();
                auto *xy = s->as<base::RealVectorStateSpace::StateType>();
                xy->values[0] = wp[0];
                xy->values[1] = wp[1];
                owned.push_back(s);
                path.push_back(s);
            }

            // setWarmStartPath copies the states, so we can free our originals immediately after.
            planner->setWarmStartPath(path);
            for (auto *s : owned)
                si->freeState(s);
        }

        /// \brief Run solve() with a zero-iteration termination condition. injectWarmStartPath() runs
        /// before the search loop, so the resulting tree contains the start plus exactly the injected
        /// subtree -- no search-added vertices or edges to pollute the counts.
        base::PlannerStatus solveZeroIterations()
        {
            base::IterationTerminationCondition itc(0u);
            return planner->solve(base::PlannerTerminationCondition(itc));
        }

        /// \brief Whether a vertex with (approximately) the given coordinates exists in the tree.
        static bool hasVertexAt(const base::PlannerData &data, double x, double y)
        {
            for (unsigned int i = 0u; i < data.numVertices(); ++i)
            {
                const auto *xy = data.getVertex(i).getState()->as<base::RealVectorStateSpace::StateType>();
                if (std::abs(xy->values[0] - x) < 1e-9 && std::abs(xy->values[1] - y) < 1e-9)
                    return true;
            }
            return false;
        }
    };
    /// \brief Seed the RNG once for the whole module. Injection and zero-iteration solves are
    /// deterministic regardless, but a fixed seed keeps the suite reproducible if it grows.
    struct SeedFixture
    {
        SeedFixture()
        {
            RNG::setSeed(1u);
        }
    };
}  // namespace

BOOST_GLOBAL_FIXTURE(SeedFixture);

// 1. A warm path entirely in free space is injected in full: the number of tree edges equals
//    path.size() - 1, and every waypoint appears as a tree vertex. Tree edges only ever come from
//    injected parent/child links here (zero search iterations, and the lone goal sample has no
//    parent), so the edge count is a direct, unpolluted measure of how much of the path was injected.
BOOST_AUTO_TEST_CASE(FreeChainFullyInjected)
{

    const std::vector<double> start{1.0, 1.0};
    const std::vector<double> goal{9.0, 9.0};
    // A horizontal chain at y == 1: well below the obstacle box, so every edge is free.
    const std::vector<std::vector<double>> waypoints{
        {1.0, 1.0}, {2.0, 1.0}, {3.0, 1.0}, {4.0, 1.0}, {5.0, 1.0}};

    WarmStartFixture fix(start, goal);
    fix.setWarmStartPath(waypoints);
    fix.solveZeroIterations();

    base::PlannerData data(fix.si);
    fix.planner->getPlannerData(data);

    // All path.size()-1 edges injected.
    BOOST_CHECK_EQUAL(data.numEdges(), waypoints.size() - 1u);

    // The path's shape is present in the tree: each waypoint is a tree vertex.
    for (const auto &wp : waypoints)
        BOOST_CHECK(WarmStartFixture::hasVertexAt(data, wp[0], wp[1]));
}

// 2. A warm path whose later edges cross the obstacle box is injected only up to (and not including)
//    the first infeasible edge -- the break-on-first-infeasible-edge behavior. Only the feasible
//    prefix becomes tree edges.
BOOST_AUTO_TEST_CASE(ObstacleCrossingChainInjectsPrefixOnly)
{

    const std::vector<double> start{5.0, 1.0};
    const std::vector<double> goal{9.0, 9.0};
    // A vertical chain at x == 5 climbing toward the box (y in [4, 6]).
    //   (5,1)->(5,2) free, (5,2)->(5,3) free, then (5,3)->(5,7) passes straight through the box.
    // So exactly the first two edges are feasible and should be injected; the third stops injection.
    const std::vector<std::vector<double>> waypoints{
        {5.0, 1.0}, {5.0, 2.0}, {5.0, 3.0}, {5.0, 7.0}};
    const unsigned int expectedFeasibleEdges = 2u;

    WarmStartFixture fix(start, goal);
    fix.setWarmStartPath(waypoints);
    fix.solveZeroIterations();

    base::PlannerData data(fix.si);
    fix.planner->getPlannerData(data);

    // Only the feasible prefix is injected, not the whole chain.
    BOOST_CHECK_EQUAL(data.numEdges(), expectedFeasibleEdges);
    BOOST_CHECK_LT(data.numEdges(), waypoints.size() - 1u);

    // The feasible-prefix waypoints are present; the post-obstacle waypoint is not a tree vertex.
    BOOST_CHECK(WarmStartFixture::hasVertexAt(data, 5.0, 1.0));
    BOOST_CHECK(WarmStartFixture::hasVertexAt(data, 5.0, 2.0));
    BOOST_CHECK(WarmStartFixture::hasVertexAt(data, 5.0, 3.0));
    BOOST_CHECK(!WarmStartFixture::hasVertexAt(data, 5.0, 7.0));
}

// 3. An empty or single-state warm path is a no-op: solve() runs normally and injects nothing.
BOOST_AUTO_TEST_CASE(EmptyOrSinglePointPathIsNoOp)
{

    const std::vector<double> start{1.0, 1.0};
    const std::vector<double> goal{9.0, 9.0};

    {
        WarmStartFixture fix(start, goal);
        fix.setWarmStartPath({});  // empty
        BOOST_CHECK_NO_THROW(fix.solveZeroIterations());

        base::PlannerData data(fix.si);
        fix.planner->getPlannerData(data);
        BOOST_CHECK_EQUAL(data.numEdges(), 0u);
    }

    {
        WarmStartFixture fix(start, goal);
        fix.setWarmStartPath({{1.0, 1.0}});  // single state
        BOOST_CHECK_NO_THROW(fix.solveZeroIterations());

        base::PlannerData data(fix.si);
        fix.planner->getPlannerData(data);
        BOOST_CHECK_EQUAL(data.numEdges(), 0u);
    }
}

// 4. The warm path is consumed once: a second solve() (no clear() in between) must not re-inject it.
//    If the consumed flag were ignored, the second solve would duplicate the subtree and the edge
//    count would grow.
BOOST_AUTO_TEST_CASE(WarmStartConsumedOnce)
{

    const std::vector<double> start{1.0, 1.0};
    const std::vector<double> goal{9.0, 9.0};
    const std::vector<std::vector<double>> waypoints{
        {1.0, 1.0}, {2.0, 1.0}, {3.0, 1.0}, {4.0, 1.0}, {5.0, 1.0}};

    WarmStartFixture fix(start, goal);
    fix.setWarmStartPath(waypoints);

    fix.solveZeroIterations();
    base::PlannerData firstData(fix.si);
    fix.planner->getPlannerData(firstData);
    const unsigned int edgesAfterFirst = firstData.numEdges();
    BOOST_CHECK_EQUAL(edgesAfterFirst, waypoints.size() - 1u);

    // Second solve with no clear(): injection already consumed, so nothing new is added.
    fix.solveZeroIterations();
    base::PlannerData secondData(fix.si);
    fix.planner->getPlannerData(secondData);

    BOOST_CHECK_EQUAL(secondData.numEdges(), edgesAfterFirst);
}
