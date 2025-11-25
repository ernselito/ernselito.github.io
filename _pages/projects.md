---
layout: single
title: "Projects & Research"
permalink: /projects/
excerpt: "A showcase of my projects, research, and work in Machine Learning and Data Science."
author_profile: true
classes: wide
---


Welcome to my projects portfolio!  
Here you’ll find selected work in **Machine Learning, Data Science, and Research**.  
Click on any project to learn more about the methodology, outcomes, and source code.

---

## Featured Projects

{% assign written_year = '' %}
{% assign all_posts = site.posts | default: [] %}

<div class="archive-wide">
  {% for post in all_posts %}
    {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
    {% if year != written_year %}
      {% assign written_year = year %}
      <!-- Optional: Uncomment below to display year headings -->
      
    {% endif %}
    {% include archive-single.html %}
  {% endfor %}
</div>
