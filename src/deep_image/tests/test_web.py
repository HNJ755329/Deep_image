from django.test import TestCase
from django_webtest import WebTest

class ViewTest(WebTest):
    def setUp(self):
        # self.post = Post.objects.create()
        pass

    def test_view_page(self):
        page = self.app.get('/')
        # print(page)
        self.assertEqual(len(page.forms), 1)