from argparse import ArgumentParser

from .adjust import adjust_pdf_margin_manual, adjust_pdf_margin_auto
from .merge import merge_pdfs


def main():
    argparser = ArgumentParser(description="PDF manipulation tools")

    subparsers = argparser.add_subparsers(help='sub-command help')

    def sub_command(*args, **kwds):
        """Add subparsers for each tool
        Usage::

            @sub_command('command-name', help='help message')
            def function(parser: argparse.ArgumentParser):
                parser.add_argument('-s', '--src', help='source PDF file')
                ...
                return function_to_call
        """
        def wrap(func):
            parser = subparsers.add_parser(*args, **kwds)
            to_call = func(parser)
            parser.set_defaults(func=to_call)
        return wrap

    @sub_command('adjust', help='adjust PDF files')
    def _(parser):
        parser.add_argument('-s', '--src', help='source PDF file', required=True)
        parser.add_argument('-d', '--dst', help='destination PDF file', required=True)
        parser.add_argument('--x-threshold', type=float, default=0.05, help='x-threshold for auto adjust')
        parser.add_argument('--skips', type=str, help='pages to skip in form like 1,3-5,7')
        return adjust_pdf_margin_auto
    
    @sub_command('adjust-page', help='adjust PDF files in pages')
    def _(parser):
        parser.add_argument('-s', '--src', help='source PDF file', required=True)
        parser.add_argument('-d', '--dst', help='destination PDF file', required=True)
        parser.add_argument('-f', '--plan-file', help='plan file', required=False)
        parser.add_argument('plan_text', nargs='*', help='plan text')
        return adjust_pdf_margin_manual

    @sub_command('merge', help='merge PDF files into one')
    def _(parser):
        parser.add_argument('input_files', nargs='+', help='paths to input files in order')
        parser.add_argument('-o', '--output', dest='output_file', required=True, help='path to output file')
        return merge_pdfs

    @sub_command('help')
    def _(parser):
        return argparser.print_help

    args = argparser.parse_args()

    try:
        func = vars(args).pop('func')
    except KeyError:
        argparser.print_help()
        return
    
    try:
        func(**vars(args))
    except Exception:
        raise


if __name__ == "__main__":
    main()