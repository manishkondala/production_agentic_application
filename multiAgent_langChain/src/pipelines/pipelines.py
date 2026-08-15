from src.agents.agents import build_search_agent, reader_agent, writer_chain, critic_chain



def run_research_pipeline(topic: str) -> dict:
    state = {}

    # Search Agent <-------------------------
    print("\n"+" ="*50)
    print("step 1 - Search agent is working...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": (["user", f"Find recent, reliable and detailed information about the asked topic"])
    })
    state["search_results"] = search_result["messages"][-1].content

    print("\n Search result ", state['search_results'])

    # Reader Agent <-------------------------
    print("\n" + " ="*50)
    print("Step 2 - Reader agent is working...")
    print("="*50)

    reader = reader_agent()
    reader_result = reader.invoke({
        "messages" : [("user",
            f"Based on the following search results about '{topic}', "
            f"Pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results: \n{state['search_results'][:800]}"
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content

    print("\nScrapped Content: \n", state['scraped_content'])


    # Writer Chain <----------------------------

    print("\n" + " ="*50)
    print("Step 3 - Writer is drafting the report...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n{state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state['report'] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n", state['report'])

    # Critic Chain <----------------------------

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n critic report \n", state['feedback'])

    return state